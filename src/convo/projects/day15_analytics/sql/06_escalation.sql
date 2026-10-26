-- Escalation rate, and how deep into a conversation it happens.
--
-- The depth matters more than the rate. A handoff on turn one is a routing
-- decision; a handoff on turn four means the customer spent three turns
-- failing first, and only the second kind is a problem worth money.
SELECT
  COUNT(DISTINCT conversation_id)                              AS conversations,
  COUNT(DISTINCT IF(escalated, conversation_id, NULL))         AS escalated,
  ROUND(SAFE_DIVIDE(
    COUNT(DISTINCT IF(escalated, conversation_id, NULL)),
    COUNT(DISTINCT conversation_id)), 4)                       AS escalation_rate,
  ROUND(AVG(IF(escalated, turn_index, NULL)), 2)               AS mean_turn_when_escalated,
  MAX(IF(escalated, turn_index, NULL))                         AS deepest_escalation
FROM {table}
WHERE source_day = 13
