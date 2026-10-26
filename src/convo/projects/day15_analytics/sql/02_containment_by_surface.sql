-- The same two numbers, split by surface.
--
-- `surface` is in the schema for this query. It is the field most likely to
-- be left out when the table is designed, because at design time there is
-- only one channel -- and it cannot be added later, since the rows are
-- already written.
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
    MAX(t.surface)                  AS surface,
    MAX(CAST(t.escalated AS INT64)) AS escalated,
    MAX(CAST(l.answered  AS INT64)) AS ended_answered
  FROM {table} AS t
  JOIN last_turn AS l USING (conversation_id)
  GROUP BY t.conversation_id
)
SELECT
  surface,
  COUNT(*)                                                       AS conversations,
  ROUND(SAFE_DIVIDE(COUNTIF(escalated = 0), COUNT(*)), 4)        AS containment_naive,
  ROUND(SAFE_DIVIDE(
    COUNTIF(escalated = 0 AND ended_answered = 1), COUNT(*)), 4)  AS containment_honest
FROM per_conversation
GROUP BY surface
ORDER BY conversations DESC
