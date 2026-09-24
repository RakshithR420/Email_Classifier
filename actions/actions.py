"""Decide what to do with a classified email, then do it.

decide()        -> pure function, returns a plan (easy to test, used for dry runs)
execute_plan()  -> performs the plan through the Gmail API
decide_and_act()-> both, plus a printed summary. Returns the plan for logging.
"""

import os

import yaml

_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config"
)

_actions_config = None


def get_actions_config():
    global _actions_config
    if _actions_config is None:
        path = os.path.join(_CONFIG_DIR, "actions.yaml")
        with open(path, "r", encoding="utf-8") as f:
            _actions_config = yaml.safe_load(f) or {}
    return _actions_config


def decide(category, confidence=1.0, valid=True, confidence_threshold=0.0,
           low_confidence_label=None, actions_config=None):
    """Returns a plan dict: {"action", "label", "reason"}.

    Confidence gating: if the model failed to answer properly (valid=False)
    or its confidence is below the threshold, we do NOT perform the
    category's action. We only add the review label (if configured) so a
    human can check it. Nothing is archived on a guess."""
    actions_config = actions_config if actions_config is not None else get_actions_config()

    if not valid or (confidence is not None and confidence < confidence_threshold):
        why = "model gave no valid answer" if not valid else (
            f"confidence {confidence:.2f} < threshold {confidence_threshold:.2f}"
        )
        if low_confidence_label:
            return {"action": "review", "label": low_confidence_label, "reason": why}
        return {"action": "none", "label": None, "reason": why}

    config = actions_config.get(category) or {"action": "none"}
    action = config.get("action", "none")
    label = config.get("label")

    if action not in ("label", "archive", "none"):
        return {"action": "none", "label": None, "reason": f"unknown action '{action}' in actions.yaml"}
    if action == "label" and not label:
        return {"action": "none", "label": None, "reason": "action is 'label' but no label is set"}
    return {"action": action, "label": label, "reason": f"category {category}"}


def execute_plan(service, email_id, plan, processed_label=None):
    """Carries out `plan` on the email through the Gmail API. Everything is
    done in one modify call per email."""
    from auth.gmail_client import get_or_create_label, modify_labels

    add, remove = [], []
    if plan["action"] in ("label", "archive", "review") and plan.get("label"):
        add.append(get_or_create_label(service, plan["label"]))
    if plan["action"] == "archive":
        remove.append("INBOX")
    if processed_label:
        add.append(get_or_create_label(service, processed_label))

    if add or remove:
        modify_labels(service, email_id, add=add, remove=remove)


def decide_and_act(service, email, category, dry_run=True, confidence=1.0, valid=True,
                   confidence_threshold=0.0, low_confidence_label=None, processed_label=None):
    """Decides what to do with `email` and either prints it (dry_run=True)
    or actually changes Gmail (dry_run=False). Returns the plan."""
    plan = decide(category, confidence, valid, confidence_threshold, low_confidence_label)
    subject = email.get("subject") or "(no subject)"
    label_txt = f" '{plan['label']}'" if plan.get("label") else ""

    if dry_run:
        if plan["action"] == "none":
            print(f"  [DRY RUN] no action ({plan['reason']})")
        else:
            print(f"  [DRY RUN] would {plan['action']}{label_txt} ({plan['reason']})")
        return plan

    execute_plan(service, email["id"], plan, processed_label=processed_label)
    if plan["action"] == "none":
        print(f"  [LIVE] no action ({plan['reason']}) -> {subject}")
    else:
        print(f"  [LIVE] {plan['action']}{label_txt} -> {subject}")
    return plan
