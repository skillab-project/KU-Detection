#!/bin/bash
set -e

echo "==================================="
echo "Starting database seeding..."
echo "==================================="

SEED_URL="https://huggingface.co/datasets/nnikolaidis/skillab-ku-analysis-2/resolve/main/seed_data.sql"
SEED_FILE="/tmp/seed_data.sql"

# Εγκατάσταση curl αν δεν υπάρχει
if ! command -v curl &> /dev/null; then
    echo "Installing curl..."
    apt-get update -qq && apt-get install -y -qq curl
fi

# Download seed file
echo "⬇️  Downloading seed data from Hugging Face..."
curl -L -o "$SEED_FILE" "$SEED_URL"

if [ ! -f "$SEED_FILE" ]; then
    echo "❌ Failed to download seed file!"
    exit 1
fi

echo "✅ Download complete. File size: $(du -h $SEED_FILE | cut -f1)"

# Run the seed file
echo "Loading data into database..."
psql -v ON_ERROR_STOP=0 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -f "$SEED_FILE"

# Διαγραφή apache repos και των σχετικών εγγραφών
echo "Removing apache repositories (keeping only eclipse)..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    BEGIN;
    DELETE FROM analysis_results WHERE repo_name IN (SELECT name FROM repositories WHERE organization = 'apache');
    DELETE FROM commits WHERE repo_name IN (SELECT name FROM repositories WHERE organization = 'apache');
    DELETE FROM repositories WHERE organization = 'apache';
    COMMIT;
EOSQL

# Καθαρισμός
rm -f "$SEED_FILE"

echo "==================================="
echo "✅ Seeding complete!"
echo "==================================="

psql --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "SELECT organization, COUNT(*) FROM repositories GROUP BY organization;"