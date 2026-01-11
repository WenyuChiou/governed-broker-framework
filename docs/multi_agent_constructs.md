# Design Document: Multi-Agent Social Constructs (The "5 Pillars")

To simulate realistic social dynamics and policy impacts, the framework adopts 5 core psychological constructs. These act as "Lego bricks" that provide standardized reasoning inputs for the LLM agents.

## 1. Threat Appraisal (TP) - *Perceived Risk*
*   **Definition**: How severe the threat feels and how vulnerable the agent perceives themselves to be.
*   **Data Sources**: Flood depth, past flood frequency, "increasing trend" rumors.
*   **Impact**: High TP accelerates adaptation (Insurance/Elevation); Extreme TP triggers Relocation.

## 2. Coping Appraisal (CP) - *Ability to Act*
*   **Definition**: Response efficacy (does it work?) and Self-efficacy (can I afford/do it?).
*   **Data Sources**: Household income, government subsidy rates, skill costs.
*   **Impact**: Low CP blocks expensive actions like Elevation, even if TP is High.

## 3. Stakeholder Trust (SP) - *Institutional Confidence*
*   **Definition**: Confidence in the reliability of insurance companies and government agencies.
*   **Data Sources**: Payout speed/success (from Memory), subsidy availability.
*   **Impact**: High SP increases insurance uptake; Low SP leads to "Do Nothing" or non-institutional adaptation (Elevation).

## 4. Social Capital (SC) - *Community Influence*
*   **Definition**: Influence of neighbors' actions and community norms.
*   **Data Sources**: Social Observation (e.g., "30% of neighbors have elevated").
*   **Impact**: Peer pressure logic—agents are more likely to adapt if they observe successful neighbors doing the same.

## 5. Place Attachment (PA) - *Emotional Bond*
*   **Definition**: Emotional connection to the community and past investment in the home.
*   **Data Sources**: Tenure (years in community), existing adaptations (sunk cost logic).
*   **Impact**: High PA significantly delays Relocation, even under moderate flood threat.

---

## Implementation Status
*   **Single-Agent Phase**: TP and CP are fully implemented.
*   **Multi-Agent Phase**: SP, SC, and PA will be integrated via the `SocialModule` and `InteractionHub` in upcoming releases.
