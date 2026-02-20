Information about the dataset: 

The 2020 Notification Bandit Dataset from Duolingo contains user-level logs of push notifications sent to users during a ~35-day period. Each row represents a single notification event and includes information about which notification templates were eligible, which template was selected, the user's past interaction history, and whether the user engaged with the lesson following the notification. 

The dataset is designed for offline evaluation of multi-armed bandit algorithms that optimize notification selection to increase user engagement. It provides a rich set of features for studying the effect of notification strategies across different languages and user contexts.

The dataset is structured in parquet files, with following format: 

	datetime - f64	ui_language - str	eligible_templates - list[str]	history - list[struct[2]]	                        selected_template - str	session_end_completed - bool
	0.153461	en	                ["G", "E", … "D"]	        [{"A",28.195648}, {"C",27.193529}, … {"F",2.001328}]	B	                false
        2.827303	es	                ["G", "E", … "D"]	        [{"A",29.836182}, {"E",28.835896}, … {"L",1.031443}]	A	                true


Here: 

| Column                  | Type              | Meaning                                                                                                                                                                                 |
| ----------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datetime`              | `f64`             | Timestamp in **floating-point seconds** since some reference (likely start of experiment). Can be used to compute time intervals and session windows.                                   |
| `ui_language`           | `str`             | User interface language (e.g., `'en'`, `'es'`). Often used for stratifying users or controlling for language effects.                                                                   |
| `eligible_templates`    | `list[str]`       | List of **notification templates that were eligible** to be sent to this user at this moment. These are the “arms” that the bandit algorithm could choose from.                                   |
| `history`               | `list[struct[2]]` | Each struct is `(template, reward)` representing **previous notifications sent and their outcomes** for the user. Used to model recency and cumulative reward for the bandit algorithm.   |
| `selected_template`     | `str`             | The **actual template** that was sent for this notification event. This is the “action” taken by the bandit.                                                                            |
| `session_end_completed` | `bool`            | Whether the user **completed a lesson within the 2-hour window** after receiving the notification. This is the **reward label** for evaluating bandit performance.                      |

How to read? 
Row 1
	0.153461  en  ["G", "E", … "D"]  [{"A",28.195648}, {"C",27.193529}, … {"F",2.001328}]  B  false

datetime: 0.153461 → very early in the experiment.
ui_language: en → user interface is English.
eligible_templates: ["G", "E", … "D"] → many templates available; the bandit algorithm had several arms to choose from.
history: [{"A",28.195648}, {"C",27.193529}, … {"F",2.001328}] → previous notifications with associated reward scores (likely cumulative or discounted reward per template for this user).
selected_template: B → this is the template actually sent.
session_end_completed: false → user did not engage within 2 hours.

Row 2
	2.827303  es  ["G", "E", … "D"]  [{"A",29.836182}, {"E",28.835896}, … {"L",1.031443}]  A  true


datetime: 2.827303 → later in the timeline.
ui_language: es → Spanish interface.
eligible_templates: same idea as above.
history: similar, but with different previous rewards.
selected_template: A → sent template.
session_end_completed: true → user did engage, so reward = 1.


But what are these eligible_templates e.g., ["G", "E", … "D"] exactly? 
These letters (A, B, C, …, L, etc.) are identifiers for specific notification templates.
Each letter corresponds to a different notification “message” that could be sent.

Example:
A → “Time for your Spanish lesson!”
B → “Don’t break your streak today!”
C → “You’re almost there, keep going!”
…and so on.

The list contains only the templates that are eligible for this user at this moment. Some templates might be restricted: e.g., only shown if streak > 3, or only at certain hours.

Key idea: The bandit algorithm can only choose from the eligible_templates for a user; the rest are unavailable for that decision point.

What about history (e.g., [{"A",28.195648}, {"C",27.193529}, … {"F",2.001328}])? 
Each element of history is a struct: (template, reward_value).
	- template → same kind of letter (A, B, C…) representing a notification template.
	- reward_value → a floating-point number, representing the past performance of that template for this user.

		-Meaning of the reward values

			These numbers (e.g., 28.195648, 2.001328) are not binary 0/1 like session_end_completed, but likely a smoothed or weighted score derived from past outcomes.

			They encode things like:

			- Cumulative past rewards: sum of previous “successes” (or completion probabilities) for this template.
			- Recency-weighted score: recent completions count more than older ones.
			- Decay/recovery effect: some templates lose effectiveness if overused — the bandit algorithm tracks this with these values.

			Example: [{"A", 28.195648}, {"C", 27.193529}, {"F", 2.001328}]
				- A = 28.195648 → Template A has been highly successful recently (high reward history).
				- C = 27.193529 → Template C also worked well, slightly less than A.
				- F = 2.001328 → Template F has almost no recent success for this user.

While the dataset is primarily designed for evaluating bandit algorithms, it can also be used for other types of data science analyses. Think out of the box, and impress us with your work during the datathon pitch! 