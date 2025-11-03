from sqlalchemy.orm import Session
from typing import List, Dict
from database import PYQ  # Consistent import

def get_pyqs_by_subject(db: Session, subject: str) -> List[PYQ]:
    """
    Retrieve all PYQ entries for the given subject.
    """
    return db.query(PYQ).filter(PYQ.subject == subject).all()

def store_pyqs(db: Session, pyqs: List[Dict], subject: str) -> int:
    """
    Store a list of PYQs in the database under the given subject.
    Returns the number of successfully inserted records.
    """
    pyq_objects = []
    for entry in pyqs:
        # Basic validation
        question = entry.get("question")
        if not question:
            continue  # skip invalid entries

        pyq_obj = PYQ(  # Use imported PYQ directly
            subject=subject,
            sub_topic=entry.get("sub_topic", ""),
            question=question,
            marks=entry.get("marks", 0),
            year=entry.get("year", ""),
            unit=entry.get("unit"),
            semester=entry.get("semester"),
            branch=entry.get("branch"),
            college=entry.get("college","IGDTUW"),
            course=entry.get("course","B.Tech"),
            frequency=1
        )
        pyq_objects.append(pyq_obj)

    try:
        db.add_all(pyq_objects)
        db.commit()
        return len(pyq_objects)
    except Exception as e:
        db.rollback()
        print(f"Error storing PYQs: {e}")
        return 0
