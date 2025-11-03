# Fixed data_loader.py
import os
import json
from database import SessionLocal, PYQ  # Import PYQ from database, not crud
import crud

def load_pyqs_from_json(json_path: str, subject: str):
    """Load PYQs from JSON file and store in database."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            pyqs = json.load(f)
    except Exception as e:
        print(f"❌ Error reading JSON file {json_path}: {e}")
        return 0

    if not pyqs:
        print(f"⚠️ No data found in {json_path}")
        return 0

    # EXTRACT METADATA FROM FILENAME (optional)
    filename = os.path.basename(json_path)
    
    # Map JSON fields to database fields
    for pyq in pyqs:
        # Map "topic" to "sub_topic" for database compatibility
        if "topic" in pyq and "sub_topic" not in pyq:
            pyq["sub_topic"] = pyq["topic"]

    try:
        with SessionLocal() as db:
            inserted_count = crud.store_pyqs(db, pyqs, subject)
        
        if inserted_count > 0:
            print(f"✅ Successfully inserted {inserted_count} PYQs for subject: {subject}")
            if inserted_count < len(pyqs):
                print(f"⚠️ Note: {len(pyqs) - inserted_count} records were skipped due to validation issues")
        else:
            print(f"❌ No PYQs were inserted for {subject}")
            
        return inserted_count
            
    except Exception as e:
        print(f"❌ Database error for {subject}: {e}")
        return 0

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    subjects_dir = os.path.join(current_dir, "subjects")

    if not os.path.exists(subjects_dir):
        print(f"❌ 'subjects' folder not found at: {subjects_dir}")
        exit(1)

    json_files = [f for f in os.listdir(subjects_dir) if f.endswith(".json")]
    if not json_files:
        print("❌ No JSON files found in the 'subjects' folder.")
        exit(1)

    print(f"📂 Found {len(json_files)} JSON files in subjects folder")
    total_inserted = 0

    for filename in json_files:
        subject_name = os.path.splitext(filename)[0]
        path = os.path.join(subjects_dir, filename)
        print(f"\n📥 Loading {filename} for subject: {subject_name}")
        
        # Load and get actual inserted count
        before_count = 0
        try:
            with SessionLocal() as db:
                before_count = db.query(PYQ).filter(PYQ.subject == subject_name).count()
        except:
            pass
            
        inserted_count = load_pyqs_from_json(path, subject_name)
        total_inserted += inserted_count
        
        # Verify actual database count after insertion (optional verification)
        try:
            with SessionLocal() as db:
                after_count = db.query(PYQ).filter(PYQ.subject == subject_name).count()
                actual_inserted = after_count - before_count
                print(f"📊 Verification: {actual_inserted} records actually added to database")
        except:
            print(f"📊 Function reported: {inserted_count} records inserted")

    print(f"\n🎉 Data loading complete! Total PYQs inserted: {total_inserted}")
