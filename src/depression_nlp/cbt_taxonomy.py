# Define the shared CBT intervention taxonomy

INTERVENTION_SUBDOMAINS = (
    "guided_discovery",
    "formulation",
    "cognitive_change",
    "behavioral_change",
    "collaboration",
    "empathy_understanding",
    "praise_encouragement",
    "agenda_goal_setting",
    "homework_review",
    "homework_assignment",
    "feedback",
    "summary_transition",
)

INTERVENTION_DOMAINS = ("domain_1", "domain_2", "domain_3")
OTHER_LABEL = "other"

SUBDOMAIN_TO_DOMAIN = {
    "guided_discovery": "domain_1",
    "formulation": "domain_1",
    "cognitive_change": "domain_1",
    "behavioral_change": "domain_1",
    "collaboration": "domain_2",
    "empathy_understanding": "domain_2",
    "praise_encouragement": "domain_2",
    "agenda_goal_setting": "domain_3",
    "homework_review": "domain_3",
    "homework_assignment": "domain_3",
    "feedback": "domain_3",
    "summary_transition": "domain_3",
    "other": "other",
}

LABEL_ALIASES = {
    "guided_discovery_socratic_questioning": "guided_discovery",
    "key_cognitions_and_behaviors_formulation": "formulation",
    "cognitive_change_methods": "cognitive_change",
    "behavioral_change_methods": "behavioral_change",
    "therapeutic_empathy_understanding": "empathy_understanding",
    "therapeutic_praise_encouragement": "praise_encouragement",
    "homework_assignment_action_planning": "homework_assignment",
    "summary_session_transition": "summary_transition",
    "other": "other",
}

ALL_SUBDOMAIN_LABELS = INTERVENTION_SUBDOMAINS + (OTHER_LABEL,)
ALL_DOMAIN_LABELS = INTERVENTION_DOMAINS + (OTHER_LABEL,)