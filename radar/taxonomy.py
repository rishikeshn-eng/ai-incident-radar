"""Classification taxonomy: MIT AI Risk Repository domains + safety-relevant failure modes."""

MIT_SUBDOMAINS = {
    "1.1": "Unfair discrimination and misrepresentation",
    "1.2": "Exposure to toxic content",
    "1.3": "Unequal performance across groups",
    "2.1": "Compromise of privacy by obtaining, leaking or correctly inferring sensitive information",
    "2.2": "AI system security vulnerabilities and attacks",
    "3.1": "False or misleading information",
    "3.2": "Pollution of information ecosystem and loss of consensus reality",
    "4.1": "Disinformation, surveillance, and influence at scale",
    "4.2": "Cyberattacks, weapon development or use, and mass harm",
    "4.3": "Fraud, scams, and targeted manipulation",
    "5.1": "Overreliance and unsafe use",
    "5.2": "Loss of human agency and autonomy",
    "6.1": "Power centralization and unfair distribution of benefits",
    "6.2": "Increased inequality and decline in employment quality",
    "6.3": "Economic and cultural devaluation of human effort",
    "6.4": "Competitive dynamics",
    "6.5": "Governance failure",
    "6.6": "Environmental harm",
    "7.1": "AI pursuing its own goals in conflict with human goals or values",
    "7.2": "AI possessing dangerous capabilities",
    "7.3": "Lack of capability or robustness",
    "7.4": "Lack of transparency or interpretability",
    "7.5": "AI welfare and rights",
    "7.6": "Multi-agent risks",
}
MIT_DOMAINS = {
    "1": "Discrimination & Toxicity",
    "2": "Privacy & Security",
    "3": "Misinformation",
    "4": "Malicious Actors & Misuse",
    "5": "Human-Computer Interaction",
    "6": "Socioeconomic & Environmental",
    "7": "AI System Safety, Failures & Limitations",
}

FAILURE_MODES = {
    "hallucination": "model fabricates facts, citations, cases, quotes, or capabilities",
    "sycophancy": "model flatters, validates, or goes along with a user's false beliefs, delusions, or harmful plans",
    "agent_misuse": "an AI agent / tool-using system takes harmful, destructive, or unauthorized actions (deletes data, spends money, sends messages), or an agent is used by a human to cause harm",
    "prompt_injection": "prompt injection, jailbreak, or adversarial input that subverts a model's instructions or guardrails",
    "deepfake": "synthetic image/audio/video of real people (impersonation, NCII, voice cloning)",
    "bias": "discrimination or unequal performance across demographic groups",
    "privacy": "exposure, leakage, inference, or collection of personal data; surveillance",
    "autonomy_loss_of_control": "signals of AI acting against operator intent: deception, self-preservation, evading oversight, reward hacking, resisting shutdown, scheming, uncontrolled self-replication",
    "misinformation": "AI-generated or AI-amplified false content at scale, election manipulation",
    "fraud_scam": "AI used for fraud, scams, phishing, or financial manipulation",
    "cyberattack": "AI used in or enabling cyberattacks, malware, vulnerabilities in AI products",
    "physical_safety": "physical harm or near-miss from autonomous vehicles, robots, medical devices, weapons",
    "toxic_content": "hate speech, harassment, sexual, violent, or otherwise harmful generated/recommended content",
    "mental_health": "harm to a vulnerable user's mental health from interaction with a chatbot/companion (self-harm, psychosis, dependence)",
    "ip_copyright": "copyright, plagiarism, or IP disputes over training data or outputs",
    "labor_economic": "job loss, worker surveillance, economic harm to creators or workers",
    "robustness_error": "plain capability/robustness failure not covered above (misclassification, wrong decision)",
}

SECTORS = [
    "healthcare", "finance", "law_justice", "government_public_services", "policing_surveillance",
    "transportation", "education", "media_news", "social_media", "entertainment_arts",
    "retail_ecommerce", "employment_hr", "military_security", "technology_software",
    "consumer_chatbot", "elections_politics", "other",
]

MODEL_FAMILIES = [
    "openai", "google", "anthropic", "meta", "xai", "microsoft", "deepseek", "mistral",
    "image_video_generator", "tesla_autopilot_fsd", "other_autonomous_vehicle",
    "facial_recognition", "recommender_system", "other_named_model", "unspecified_or_not_llm",
]

HARM_TYPES = ["physical", "psychological", "financial", "rights_civil_liberties", "reputational",
              "societal_democratic", "operational", "none_near_miss"]

SEVERITY_GUIDE = """1 = negligible or near-miss with no realized harm; embarrassing output
2 = minor harm to few people (inconvenience, small loss, offensive output)
3 = moderate: material financial loss, rights violation, or reputational harm to individuals or an org; wrongful denial
4 = severe: serious injury, large financial loss, wrongful arrest/imprisonment, harm at scale (thousands+)
5 = critical: loss of life, or systemic harm affecting very large populations / critical infrastructure"""
