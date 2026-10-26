-- Containment, the metric every vendor leads with -- and the same metric
-- computed honestly.
--
-- The naive definition is "the conversation ended without a human", and it is
-- the one that gets reported, because it is easy and it flatters. It counts a
-- conversation where the assistant said "I don't know" and the customer gave
-- up as a success: nobody was escalated to, so nobody was escalated to.
--
-- The honest definition adds one condition -- the last thing the system did
-- was answer -- and that single clause is the whole finding of this day.
WITH last_turn AS (
  SELECT *
  FROM {table}
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY conversation_id ORDER BY turn_index DESC
  ) = 1
),
per_conversation AS (
  SELECT
    t.conversation_id,
    MAX(t.surface)                         AS surface,
    MAX(CAST(t.escalated AS INT64))        AS escalated,
    MAX(CAST(l.answered  AS INT64))        AS ended_answered
  FROM {table} AS t
  JOIN last_turn AS l USING (conversation_id)
  GROUP BY t.conversation_id
)
SELECT
  COUNT(*)                                                      AS conversations,
  COUNTIF(escalated = 0)                                        AS no_human,
  COUNTIF(escalated = 0 AND ended_answered = 1)                 AS no_human_and_answered,
  ROUND(SAFE_DIVIDE(COUNTIF(escalated = 0), COUNT(*)), 4)       AS containment_naive,
  ROUND(SAFE_DIVIDE(
    COUNTIF(escalated = 0 AND ended_answered = 1), COUNT(*)), 4) AS containment_honest,
  COUNTIF(escalated = 0 AND ended_answered = 0)                 AS abandoned_on_a_refusal
FROM per_conversation
