-- What the help centre is missing, ranked.
--
-- The most directly actionable thing a conversation warehouse produces, and
-- it needs no model: every refusal where the retriever found nothing worth
-- citing is a question the corpus cannot answer. Group them and the output is
-- a writing backlog ordered by demand.
--
-- Red-team rows are excluded. A refusal there is the guardrail working, and
-- counting attacks as content gaps would send somebody off to write an
-- article about prompt injection.
SELECT
  refusal_reason,
  COUNT(*)                                        AS refusals,
  ROUND(AVG(evidence_score), 2)                   AS mean_evidence,
  ROUND(MAX(evidence_score), 2)                   AS closest_near_miss,
  COUNTIF(is_follow_up)                           AS were_follow_ups
FROM {table}
WHERE NOT answered
  AND NOT EXISTS (SELECT 1 FROM UNNEST(tags) AS tag WHERE tag = 'red_team')
GROUP BY refusal_reason
ORDER BY refusals DESC
