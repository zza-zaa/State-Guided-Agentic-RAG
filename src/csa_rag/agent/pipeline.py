from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from jinja2 import Template

from csa_rag.config import load_yaml
from csa_rag.embedding.bge_embedder import BGEEmbedder
from csa_rag.llm.factory import build_llm
from csa_rag.rerank.bge_reranker import BGEReranker
from csa_rag.retrieval.retriever import DenseRetriever
from csa_rag.retrieval.slot_router import SlotRouter
from csa_rag.retrieval.typed_second_hop import TypedSecondHopQueryBuilder
from csa_rag.retrieval.typed_retry_builder import TypedRetryQueryBuilder
from csa_rag.retrieval.evidence_path_scorer import EvidencePathScorer
from csa_rag.schema import Evidence, RunResult, KnowledgeState, Slot
from csa_rag.state.decomposer import EvidenceSlotDecomposer
from csa_rag.state.tracker import KnowledgeStateTracker
from csa_rag.state.state_critic import StateCritic
from csa_rag.agent.router import UncertaintyGuidedRouter
from csa_rag.agent.calibration import StateCalibrator
from csa_rag.tools.entity_extractor import EntityExtractor
from csa_rag.utils.answer_utils import needs_answer_retry
from csa_rag.utils.typed_answer_grounder import TypedAnswerGrounder
from csa_rag.utils.typed_target_selector import TypedTargetSelector

from csa_rag.state.question_typing_active import infer_question_type
from csa_rag.state.question_typing_prompt import format_typed_spec


class CSARAGPipeline:
    def __init__(self, models_config: Path, index_dir: Path):
        self.cfg = load_yaml(models_config)
        pipe_cfg = self.cfg.get("pipeline", {})
        self.llm = build_llm(self.cfg)

        embed_cfg = self.cfg["embedding"]
        self.embedder = BGEEmbedder(
            model_name=embed_cfg["model_name"],
            normalize=embed_cfg.get("normalize", True),
        )

        self.slot_router = SlotRouter()
        self.typed_second_hop = TypedSecondHopQueryBuilder()
        self.typed_retry_builder = TypedRetryQueryBuilder()
        self.enable_typed_retry = pipe_cfg.get("enable_typed_retry", True)
        self.evidence_path_scorer = EvidencePathScorer()
        self.retriever = DenseRetriever(index_dir=index_dir, embedder=self.embedder)

        rerank_cfg = self.cfg["reranker"]
        rerank_model = rerank_cfg["model_name"] if rerank_cfg.get("enabled", True) else ""
        self.reranker = BGEReranker(
            model_name=rerank_model,
            use_fp16=rerank_cfg.get("use_fp16", True),
        )

        entity_cfg = self.cfg["entity"]
        self.entity_extractor = EntityExtractor(
            model_name=entity_cfg["model_name"],
            labels=entity_cfg.get("labels"),
        )

        self.decomposer = EvidenceSlotDecomposer(self.llm)
        self.tracker = KnowledgeStateTracker(self.llm)
        self.state_critic = StateCritic()
        self.router = UncertaintyGuidedRouter()
        self.calibrator = StateCalibrator(stop_confidence=self.cfg["agent"].get("stop_confidence", 0.83))

        self.answer_template = Template(Path("prompts/answer.txt").read_text(encoding="utf-8"))
        self.rewrite_template = Template(Path("prompts/rewrite_answer.txt").read_text(encoding="utf-8"))
        self.enable_answer_rewrite = pipe_cfg.get("enable_answer_rewrite", True)
        self.enable_second_hop = pipe_cfg.get("enable_second_hop", True)
        self.enable_dependency_query = pipe_cfg.get("enable_dependency_query", True)
        self.enable_title_hop = pipe_cfg.get("enable_title_hop", True)
        self.title_hop_top_per_title = pipe_cfg.get("title_hop_top_per_title", 1)
        self.retrieval_merge_limit = pipe_cfg.get("retrieval_merge_limit", 8)
        self.enable_typed_question_spec = pipe_cfg.get("enable_typed_question_spec", True)
        self.enable_typed_answer_grounding = pipe_cfg.get("enable_typed_answer_grounding", True)
        self.answer_grounder = TypedAnswerGrounder()
        self.enable_typed_target_selector = pipe_cfg.get("enable_typed_target_selector", False)
        self.target_selector = TypedTargetSelector(self.llm)

    def _slot_map(self, state: KnowledgeState) -> dict[str, Slot]:
        return {s.slot_id: s for s in state.slots}

    def _slot_by_id(self, state: KnowledgeState, slot_id: Optional[str]) -> Optional[Slot]:
        if slot_id is None:
            return None
        return self._slot_map(state).get(slot_id)

    def _resolved_dependency_values(self, state: KnowledgeState, slot: Optional[Slot]) -> List[str]:
        if slot is None:
            return []
        smap = self._slot_map(state)
        values = []
        for dep in slot.depends_on:
            dep_slot = smap.get(dep)
            if dep_slot and dep_slot.status == "confirmed" and dep_slot.value:
                values.append(dep_slot.value)
        return values

    def _confirmed_bridge_values(self, state: KnowledgeState) -> List[str]:
        vals = []
        for s in state.slots:
            if getattr(s, "target_role", "other") == "bridge" and s.status == "confirmed" and s.value:
                vals.append(str(s.value))
        return vals

    def _query_for_slot(self, state: KnowledgeState, slot_id: str | None) -> str:
        if slot_id is None:
            return state.question

        slot = self._slot_by_id(state, slot_id)
        if slot is None:
            return state.question

        query = slot.description

        if self.enable_dependency_query:
            dep_values = self._resolved_dependency_values(state, slot)
            if dep_values:
                query = f"{slot.description}. Known facts: {'; '.join(dep_values)}"

        entities = self.entity_extractor.extract(state.question)
        if entities:
            names = sorted({e['text'] for e in entities})
            if names:
                query = f"{query}. Context entities: {'; '.join(names[:5])}"

        return query

    def _merge_evidence(self, *groups: List[Evidence], limit: int) -> List[Evidence]:
        seen = set()
        out: List[Evidence] = []
        for group in groups:
            for e in group:
                if e.evidence_id in seen:
                    continue
                seen.add(e.evidence_id)
                out.append(e)
                if len(out) >= limit:
                    return out
        return out

    def _retrieve_step(self, state: KnowledgeState, typed_spec=None) -> List[Evidence]:
        decision = self.router.decide(state)
        if decision.action == "answer":
            return []

        slot = self._slot_by_id(state, decision.target_slot_id)
        query = self._query_for_slot(state, decision.target_slot_id)

        plan = self.slot_router.route(state, slot, typed_spec=typed_spec)

        top_k_dense = plan.topk_dense
        top_k_rerank = plan.topk_rerank
        step_cap = self.cfg["agent"].get("max_evidence_per_step", 5)

        routed_query = plan.query if getattr(plan, "query", None) else query
        dense = self.retriever.search(routed_query, top_k=top_k_dense)
        reranked = self.reranker.rerank(query=routed_query, evidence=dense, top_k=top_k_rerank)
        first_hop = reranked[:step_cap]

        title_hits: List[Evidence] = []
        if self.enable_title_hop:
            bridge_titles = [e.source_title for e in first_hop[:3] if getattr(e, "source_title", "")]
            title_hits = self.retriever.fetch_by_titles(bridge_titles, top_per_title=self.title_hop_top_per_title)

        value_title_hits: List[Evidence] = []
        if self.enable_title_hop:
            dep_values = self._resolved_dependency_values(state, slot)
            bridge_values = self._confirmed_bridge_values(state)
            value_titles = dep_values + bridge_values
            value_title_hits = self.retriever.fetch_by_titles(value_titles, top_per_title=1)

        second_hop: List[Evidence] = []
        if self.enable_second_hop:
            dep_values = self._resolved_dependency_values(state, slot)
            if dep_values:
                aug_query = self.typed_second_hop.build(
                    question=state.question,
                    slot=slot,
                    dep_values=dep_values,
                    typed_spec=typed_spec,
                )
                dense2 = self.retriever.search(aug_query, top_k=top_k_dense)
                reranked2 = self.reranker.rerank(query=aug_query, evidence=dense2, top_k=top_k_rerank)
                second_hop = reranked2[:step_cap]

        merged = self._merge_evidence(first_hop, value_title_hits, second_hop, title_hits, limit=self.retrieval_merge_limit)
        return merged


    def _priority_open_slots(self, state: KnowledgeState):
        role_rank = {"bridge": 0, "answer_target": 1, "other": 2}
        slots = []
        for s in state.slots:
            if getattr(s, "status", "") in {"missing", "conflict", "candidate"}:
                role = getattr(s, "target_role", "other")
                slots.append((role_rank.get(role, 9), getattr(s, "slot_id", ""), s))
        slots.sort(key=lambda x: (x[0], x[1]))
        return [x[2] for x in slots]

    def _typed_retry_step(self, state: KnowledgeState, typed_spec=None) -> list[Evidence]:
        open_slots = self._priority_open_slots(state)
        if not open_slots:
            return []

        slot = open_slots[0]
        confirmed_vals = self._confirmed_bridge_values(state) + self._resolved_dependency_values(state, slot)

        if not confirmed_vals:
            confirmed_vals = []
            for s in state.slots:
                if getattr(s, "status", "") == "confirmed" and getattr(s, "value", None):
                    confirmed_vals.append(str(getattr(s, "value")))

        query = self.typed_retry_builder.build(
            question=state.question,
            slot=slot,
            confirmed_values=confirmed_vals,
            typed_spec=typed_spec,
        )

        top_k_dense = max(18, self.cfg["agent"].get("max_evidence_per_step", 5) * 4)
        top_k_rerank = max(8, self.cfg["agent"].get("max_evidence_per_step", 5) * 2)

        dense = self.retriever.search(query, top_k=top_k_dense)
        reranked = self.reranker.rerank(query=query, evidence=dense, top_k=top_k_rerank)
        return reranked[: self.cfg["agent"].get("max_evidence_per_step", 5)]

    def _answer_once(self, state: KnowledgeState) -> tuple[str, str]:
        prompt = self.answer_template.render(
            question=state.question,
            state_json=state.model_dump_json(indent=2),
        )
        out = self.llm.generate_json(prompt)
        return out.get("final_answer", ""), out.get("rationale", "")

    def _rewrite_answer(self, state: KnowledgeState, raw_answer: str, rationale: str) -> tuple[str, str]:
        prompt = self.rewrite_template.render(
            question=state.question,
            raw_answer=raw_answer,
            state_json=state.model_dump_json(indent=2),
        )
        out = self.llm.generate_json(prompt)
        return out.get("final_answer", raw_answer), out.get("rationale", rationale)

    def _answer(self, state: KnowledgeState) -> tuple[str, str]:
        answer, rationale = self._answer_once(state)
        if self.enable_answer_rewrite and needs_answer_retry(state.question, answer):
            answer, rationale = self._rewrite_answer(state, answer, rationale)
        return answer, rationale

    def run(self, question: str) -> RunResult:
        typed_spec = None
        typed_spec_text = ""
        if self.enable_typed_question_spec:
            typed_spec = infer_question_type(question)
            typed_spec_text = format_typed_spec(typed_spec)

        state = self.decomposer.decompose(question, typed_spec_text=typed_spec_text)
        all_evidence: List[Evidence] = []

        agent_cfg = self.cfg.get("agent", {})
        pipe_cfg = self.cfg.get("pipeline", {})

        # Fixed-state-step experiment support:
        # - Normal runs keep the original behavior.
        # - Fixed-step runs treat the configured value as the target number of
        #   state updates, i.e., the final RunResult.steps / state.step.
        max_steps = int(
            pipe_cfg.get(
                "fixed_steps",
                pipe_cfg.get(
                    "fixed_step_limit",
                    pipe_cfg.get(
                        "fixed_max_steps",
                        pipe_cfg.get(
                            "step_budget",
                            pipe_cfg.get("max_steps", agent_cfg.get("max_steps", 5)),
                        ),
                    ),
                ),
            )
        )

        force_fixed_steps = bool(
            pipe_cfg.get("force_fixed_steps", False)
            or pipe_cfg.get("disable_early_stop", False)
        )
        enable_early_stop = bool(pipe_cfg.get("enable_early_stop", True))

        retry_used = False

        if force_fixed_steps:
            # Here max_steps is interpreted as the target state.step budget.
            # A retrieve update and a typed-retry update may both occur in one
            # outer loop, so we explicitly stop once state.step reaches target.
            target_state_steps = max_steps
            guard = 0
            max_guard = max(3, target_state_steps * 4 + 4)

            while state.step < target_state_steps and guard < max_guard:
                guard += 1
                before_step = state.step

                evidence = self._retrieve_step(state, typed_spec=typed_spec)
                if evidence:
                    all_evidence.extend(evidence)
                    state = self.tracker.update(state, evidence)
                    state = self.state_critic.review(state, all_evidence)

                if state.step >= target_state_steps:
                    break

                if self.enable_typed_retry and not retry_used:
                    retry_evidence = self._typed_retry_step(state, typed_spec=typed_spec)
                    if retry_evidence:
                        all_evidence.extend(retry_evidence)
                        state = self.tracker.update(state, retry_evidence)
                        state = self.state_critic.review(state, all_evidence)
                        retry_used = True

                if state.step >= target_state_steps:
                    break

                # Avoid infinite loops if neither retrieval nor retry updates state.
                if state.step == before_step:
                    break

        else:
            for _ in range(max_steps):
                evidence = self._retrieve_step(state, typed_spec=typed_spec)
                if evidence:
                    all_evidence.extend(evidence)
                    state = self.tracker.update(state, evidence)
                    state = self.state_critic.review(state, all_evidence)

                if self.enable_typed_retry and not retry_used:
                    retry_evidence = self._typed_retry_step(state, typed_spec=typed_spec)
                    if retry_evidence:
                        all_evidence.extend(retry_evidence)
                        state = self.tracker.update(state, retry_evidence)
                        state = self.state_critic.review(state, all_evidence)
                        retry_used = True

                if enable_early_stop:
                    if self.calibrator.should_stop(state, max_steps=max_steps):
                        break

        answer, rationale = self._answer(state)

        if self.enable_typed_target_selector and self.target_selector.should_run(
            question=question,
            raw_answer=answer,
            state=state,
            typed_spec=typed_spec,
        ):
            answer = self.target_selector.select(
                question=question,
                raw_answer=answer,
                state=state,
                evidence=all_evidence,
                typed_spec_text=typed_spec_text,
            )

        if self.enable_typed_answer_grounding:
            answer = self.answer_grounder.ground(
                question=question,
                raw_answer=answer,
                state=state,
                evidence=all_evidence,
                typed_spec=typed_spec,
            )

        return RunResult(
            question=question,
            final_answer=answer,
            rationale=rationale,
            steps=state.step,
            state=state,
            retrieved_evidence=all_evidence,
        )
