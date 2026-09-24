from actions.actions import decide, decide_and_act, get_actions_config
from classifier.classifier import get_categories

CFG = {
    "Orders": {"action": "label", "label": "AI/Orders"},
    "Promotions": {"action": "archive", "label": "AI/Promotions"},
    "Personal": {"action": "none"},
    "Broken": {"action": "label"},
}


def test_every_category_has_an_action():
    assert set(get_categories()) == set(get_actions_config())


def test_label_archive_none():
    assert decide("Orders", actions_config=CFG) == {"action": "label", "label": "AI/Orders", "reason": "category Orders"}
    assert decide("Promotions", actions_config=CFG)["action"] == "archive"
    assert decide("Personal", actions_config=CFG)["action"] == "none"
    assert decide("Unknown", actions_config=CFG)["action"] == "none"
    assert decide("Broken", actions_config=CFG)["action"] == "none"


def test_low_confidence_goes_to_review_not_archive():
    plan = decide("Promotions", confidence=0.3, confidence_threshold=0.6,
                  low_confidence_label="AI/Needs-Review", actions_config=CFG)
    assert plan["action"] == "review" and plan["label"] == "AI/Needs-Review"


def test_invalid_model_answer_is_never_acted_on():
    plan = decide("Promotions", confidence=0.99, valid=False, confidence_threshold=0.6,
                  low_confidence_label=None, actions_config=CFG)
    assert plan["action"] == "none"


def test_dry_run_never_touches_gmail():
    class Explode:
        def __getattr__(self, name):
            raise AssertionError("Gmail API was called during a dry run")

    plan = decide_and_act(Explode(), {"id": "1", "subject": "Sale!"}, "Promotions", dry_run=True)
    assert plan["action"] == "archive"
