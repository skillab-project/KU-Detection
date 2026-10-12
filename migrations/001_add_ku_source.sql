-- ─────────────────────────────────────────────────────────────────────────
-- Migration 001: ku_source column
--
-- Adds a column to analysis_results that marks which detector produced the row:
--   'legacy'      → original 27-KU Java CodeBERT model (K1..K28)
--   'detector17'  → KU-Detector-17 UnixCoder model (MQTT, REST_APIs, ...)
--
-- Backfills existing rows as 'legacy' (they were all produced by the old pipeline).
-- Adds an index to keep per-source queries fast.
-- Idempotent — safe to run multiple times.
-- ─────────────────────────────────────────────────────────────────────────

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'analysis_results'
          AND column_name = 'ku_source'
    ) THEN
        ALTER TABLE analysis_results
            ADD COLUMN ku_source VARCHAR(32) NOT NULL DEFAULT 'legacy';

        -- Backfill (DEFAULT already covered it, but make it explicit for clarity)
        UPDATE analysis_results SET ku_source = 'legacy' WHERE ku_source IS NULL;

        CREATE INDEX IF NOT EXISTS idx_analysis_results_ku_source
            ON analysis_results (ku_source);

        CREATE INDEX IF NOT EXISTS idx_analysis_results_repo_source
            ON analysis_results (repo_name, ku_source);

        RAISE NOTICE 'Added ku_source column and indexes to analysis_results.';
    ELSE
        RAISE NOTICE 'ku_source column already exists — skipping.';
    END IF;
END $$;
