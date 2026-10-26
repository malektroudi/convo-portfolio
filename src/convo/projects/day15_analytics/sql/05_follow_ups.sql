-- Whether a turn that depended on the one before it was answered as often.
--
-- The question a multi-turn assistant has to be able to answer about itself,
-- and the reason `is_follow_up` is a column rather than something inferred
-- later from turn_index: a follow-up is a property of the utterance, not of
-- its position, and the two come apart as soon as a conversation restarts.
SELECT
  is_follow_up,
  COUNT(*)                                             AS turns,
  COUNTIF(answered)                                    AS answered,
  ROUND(SAFE_DIVIDE(COUNTIF(answered), COUNT(*)), 4)   AS answer_rate,
  COUNTIF(rewritten)                                   AS rewritten_first
FROM {table}
WHERE source_day IN (11, 13)
GROUP BY is_follow_up
ORDER BY is_follow_up
