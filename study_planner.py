# study_planner.py
from sqlalchemy.orm import sessionmaker
from database import PYQ, Syllabus, get_session
# from improved_rag_pipeline import get_relevant_pyqs_flexible, debug_filter_matching
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy import func, text
from rag_pipeline import get_relevant_pyqs_with_filters
@dataclass
class StudyPlan:
    """Data class for study plan structure"""
    subject: str
    total_days: int
    exam_date: Optional[datetime]
    difficulty_level: str  # beginner, intermediate, advanced
    focus_areas: List[str]
    daily_schedule: Dict[str, Dict]
    recommended_pyqs: List[Dict]
    unit_priorities: Dict[str, int]
    

# study_planner.py - Complete replacement file

from sqlalchemy.orm import sessionmaker
from database import PYQ, get_session, SessionLocal
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Optional

class StudyPlanGenerator:
    """Simplified study plan generator for Streamlit integration"""
    
    def __init__(self):
        """Initialize with basic configuration"""
        self.difficulty_weights = {
            'beginner': {'theory': 0.6, 'practice': 0.4},
            'intermediate': {'theory': 0.4, 'practice': 0.6},
            'advanced': {'theory': 0.3, 'practice': 0.7}
        }
    
    def create_study_plan(self, session, context, duration_days):
        """
        Create study plan with simplified interface for Streamlit app
        
        Args:
            session: Database session
            context: Dict with college, branch, subject, semester
            duration_days: Number of days for the study plan
        """
        # Extract parameters from context
        subject = context.get('subject', 'Unknown Subject')
        branch = context.get('branch')
        college = context.get('college') 
        semester = context.get('semester')
        
        try:
            # Analyze PYQ patterns using the session
            pyq_analysis = self._analyze_pyq_patterns_with_session(session, context)
            
            if pyq_analysis['total_questions'] == 0:
                return {
                    'error': f'No questions found for subject: {subject}. Check your filter settings.'
                }
            
            # Calculate questions per day
            questions_per_day = max(5, min(15, pyq_analysis['total_questions'] // duration_days))
            
            # Get topics from PYQ analysis
            topics = list(pyq_analysis['sub_topic_frequency'].keys())[:5]
            
            # Create daily targets
            daily_targets = []
            for day in range(1, duration_days + 1):
                # Get questions for this day
                day_questions = self._get_questions_for_day(
                    session, context, day, questions_per_day
                )
                
                primary_topic = topics[(day - 1) % len(topics)] if topics else "General"
                
                daily_targets.append({
                    'day': day,
                    'primary_topic': primary_topic,
                    'questions_count': len(day_questions),
                    'questions': day_questions,
                    'total_marks': sum(q.get('marks', 1) for q in day_questions),
                    'estimated_time_minutes': len(day_questions) * 20  # 20 min per question
                })
            
            # Calculate statistics
            total_questions = sum(len(target['questions']) for target in daily_targets)
            total_estimated_hours = sum(target['estimated_time_minutes'] for target in daily_targets) / 60
            
            return {
                'duration_days': duration_days,
                'total_questions': total_questions,
                'questions_per_day': questions_per_day,
                'total_estimated_hours': total_estimated_hours,
                'daily_targets': daily_targets,
                'topic_distribution': dict(pyq_analysis['sub_topic_frequency'].most_common(10)),
                'subject': subject,
                'context': context
            }
            
        except Exception as e:
            return {'error': f'Error creating study plan: {str(e)}'}

    def _analyze_pyq_patterns_with_session(self, session, context):
        """Modified version that accepts session parameter"""
        # Build query with context filters
        query = session.query(PYQ)
        
        # Apply filters from context
        for field, value in context.items():
            if hasattr(PYQ, field) and value and value not in ['All Colleges', 'All Branches', 'All Types']:
                query = query.filter(getattr(PYQ, field) == value)
        
        pyqs = query.all()
        
        analysis = {
            'total_questions': len(pyqs),
            'unit_frequency': Counter(),
            'sub_topic_frequency': Counter(),
            'marks_distribution': Counter(),
            'year_trends': Counter(),
            'semester_patterns': Counter()
        }
        
        for pyq in pyqs:
            if pyq.unit:
                analysis['unit_frequency'][pyq.unit] += getattr(pyq, 'frequency', 1)
            if pyq.sub_topic:
                analysis['sub_topic_frequency'][pyq.sub_topic] += getattr(pyq, 'frequency', 1)
            if pyq.marks:
                analysis['marks_distribution'][f"{pyq.marks} marks"] += 1
            if pyq.year:
                analysis['year_trends'][str(pyq.year)] += 1
            if hasattr(pyq, 'semester') and pyq.semester:
                analysis['semester_patterns'][pyq.semester] += 1
        
        return analysis
    
    def _get_questions_for_day(self, session, context, day, target_count):
        """Get questions for a specific day"""
        # Build base query
        query = session.query(PYQ)
        
        # Apply context filters
        for field, value in context.items():
            if hasattr(PYQ, field) and value and value not in ['All Colleges', 'All Branches', 'All Types']:
                query = query.filter(getattr(PYQ, field) == value)
        
        # Get questions with offset for variety across days
        offset = (day - 1) * target_count
        questions = query.offset(offset).limit(target_count).all()
        
        # Convert to dict format
        return [
            {
                'question': q.question,
                'marks': q.marks or 1,
                'sub_topic': q.sub_topic or 'General',
                'year': q.year or 'Unknown'
            }
            for q in questions
        ]

    def get_study_progress(self, study_plan, completed_days):
        """Calculate study progress"""
        total_days = study_plan['duration_days']
        total_questions = study_plan['total_questions']
        
        # Calculate completed questions
        questions_completed = 0
        time_spent_minutes = 0
        
        for target in study_plan['daily_targets'][:completed_days]:
            questions_completed += target['questions_count']
            time_spent_minutes += target['estimated_time_minutes']
        
        return {
            'completed_days': completed_days,
            'total_days': total_days,
            'completion_percentage': (completed_days / total_days) * 100,
            'questions_completed': questions_completed,
            'total_questions': total_questions,
            'question_progress_percentage': (questions_completed / total_questions * 100) if total_questions > 0 else 0,
            'time_spent_minutes': time_spent_minutes,
            'total_estimated_minutes': study_plan['total_estimated_hours'] * 60
        }


# Keep only the simple create_study_plan function for compatibility
def create_study_plan(session, context, duration_days):
    """
    Simple wrapper function that matches your main file's expectations
    """
    generator = StudyPlanGenerator()
    return generator.create_study_plan(session, context, duration_days)


# Remove all the complex original functions and classes
# Keep only what's needed for your Streamlit app