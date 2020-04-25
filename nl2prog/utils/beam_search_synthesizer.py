from typing \
    import List, Callable, TypeVar, Tuple, Dict, Optional, Union, cast, Generic
from dataclasses import dataclass
from nl2prog.language.evaluator import Evaluator
from nl2prog.language.ast import AST, Root
from nl2prog.language.action \
    import NodeConstraint, NodeType, ExpandTreeRule, Action, \
    ApplyRule, GenerateToken, ActionOptions, Rule, CloseNode
from nl2prog.utils import TopKElement


"""
True if the argument 0 is subtype of the argument 1
"""
IsSubtype = Callable[[Union[str, Root], Union[str, Root]], bool]
State = TypeVar("State")
Input = TypeVar("Input")


@dataclass
class LazyLogProbability:
    get_rule_prob: Callable[[], Dict[Rule, float]]
    get_token_prob: Callable[[], Dict[Union[CloseNode, str], float]]

    @property
    def rule_prob(self):
        return self.get_rule_prob()

    @property
    def token_prob(self):
        return self.get_token_prob()


@dataclass
class Hypothesis(Generic[State]):
    id: int
    parent: Optional[int]
    score: float
    evaluator: Evaluator
    state: State


@dataclass
class Progress:
    id: int
    parent: Optional[int]
    score: float
    action: Action
    is_complete: bool


@dataclass
class Candidate:
    score: float
    ast: AST


class BeamSearchSynthesizer(Generic[State]):
    def __init__(self, beam_size: int,
                 is_subtype: IsSubtype, options=ActionOptions(True, True),
                 max_steps: Optional[int] = None):
        """
        Parameters
        ----------
        beam_size: int
            The number of candidates
        is_subtype: IsSubType
            The function to check the type relations between 2 node types.
            This returns true if the argument 0 is subtype of the argument 1.
        options: ActionOptions
        max_steps: Optional[int]
        """
        self._beam_size = beam_size
        self._is_subtype = is_subtype
        self._options = options
        self._max_steps = max_steps

    def initialize(self, input: Input) -> State:
        raise NotImplementedError

    def batch_update(self, hs: List[Hypothesis]) \
            -> List[Tuple[State, LazyLogProbability]]:
        raise NotImplementedError

    def load_state_dict(self, state_dict: Dict[str, object]):
        raise NotImplementedError

    def state_dict(self) -> Dict[str, object]:
        raise NotImplementedError

    def synthesize(self, input: Input):
        """
        Synthesize the program from the query

        Parameters
        ----------
        input: Input

        Yields
        ------
        candidates: List[Candidate]
            The candidate of AST
        progress: List[Progress]
            The progress of synthesizing.
        """
        candidates: List[Candidate] = []
        n_ids = 0

        # Create initial hypothesis
        evaluator = Evaluator(self._options)
        # Add initial rule
        evaluator.eval(ApplyRule(
            ExpandTreeRule(NodeType(Root(), NodeConstraint.Node),
                           [("root", NodeType(Root(), NodeConstraint.Node))])))
        state = self.initialize(input)
        hs: List[Hypothesis] = \
            [Hypothesis(0, None, 0.0, evaluator,
                        state)]
        n_ids += 1

        steps = 0
        while len(candidates) < self._beam_size:
            if self._max_steps is not None:
                if steps > self._max_steps:
                    break
                steps += 1

            # Create batch of hypothesis
            results = self.batch_update(hs)
            elem_size = self._beam_size - len(candidates)
            topk = TopKElement(elem_size)
            for i, (h, (state, lazy_prob)) in enumerate(zip(hs, results)):
                # Create hypothesis from h
                head = h.evaluator.head
                is_token = False
                if head is None:
                    continue
                head_field = \
                    cast(ExpandTreeRule,
                         cast(ApplyRule,
                              h.evaluator.action_sequence.sequence[head.action]
                              ).rule
                         ).children[head.field][1]
                if head_field.constraint == NodeConstraint.Token:
                    is_token = True
                if is_token:
                    # Generate token
                    log_prob_token = lazy_prob.token_prob
                    for token, log_prob in log_prob_token.items():
                        topk.add(h.score + log_prob, (i, GenerateToken(token)))
                else:
                    # Apply rule
                    log_prob_rule = lazy_prob.rule_prob
                    for rule, log_prob in log_prob_rule.items():
                        action = ApplyRule(rule)
                        if isinstance(action.rule, ExpandTreeRule):
                            if self._options.retain_variadic_fields:
                                if head_field.type_name == Root() and \
                                    (action.rule.parent.constraint !=
                                        NodeConstraint.Variadic) and \
                                    (action.rule.parent.type_name !=
                                        Root()):
                                    topk.add(h.score + log_prob,
                                             (i, action))
                                elif action.rule.parent.type_name != \
                                    Root() and \
                                    self._is_subtype(
                                        action.rule.parent.type_name,
                                        head_field.type_name):
                                    topk.add(h.score + log_prob,
                                             (i, action))
                            else:
                                if head_field.type_name == Root() and \
                                    (action.rule.parent.type_name !=
                                        Root()) and \
                                    (action.rule.parent.constraint !=
                                        NodeConstraint.Variadic):
                                    topk.add(h.score + log_prob,
                                             (i, action))
                                elif action.rule.parent.type_name != \
                                    Root() and \
                                    ((head_field.constraint ==
                                        NodeConstraint.Variadic) or
                                     (action.rule.parent.constraint ==
                                        NodeConstraint.Variadic)):
                                    if action.rule.parent == \
                                            head_field:
                                        topk.add(h.score + log_prob,
                                                 (i, action))
                                elif action.rule.parent.type_name != \
                                    Root() and \
                                    self._is_subtype(
                                        action.rule.parent.type_name,
                                        head_field.type_name):
                                    topk.add(h.score + log_prob,
                                             (i, action))
                        else:
                            # CloseVariadicFieldRule
                            if self._options.retain_variadic_fields and \
                                head_field is not None and \
                                head_field.constraint == \
                                    NodeConstraint.Variadic:
                                topk.add(h.score + log_prob,
                                         (i, action))

            # Instantiate top-k hypothesis
            hs_new = []
            cs = []
            ps = []
            for score, (i, action) in topk.elements:
                h = hs[i]
                state = results[i][0]
                id = n_ids
                n_ids += 1
                evaluator = h.evaluator.clone()
                evaluator.eval(action)

                if evaluator.head is None:
                    # Complete
                    c = Candidate(score, evaluator.generate_ast())
                    cs.append(c)
                    candidates.append(c)
                    ps.append(Progress(id, h.id, score,
                                       evaluator.action_sequence.sequence[-1],
                                       True))
                else:
                    hs_new.append(Hypothesis(
                        id, h.id, score, evaluator,
                        state))
                    ps.append(Progress(id, h.id, score,
                                       evaluator.action_sequence.sequence[-1],
                                       False))

            hs = hs_new
            yield cs, ps
            if len(hs) == 0:
                break
