#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Constants
NUM_BOOKS = 2000
START_DATE = datetime(2010, 1, 1)
END_DATE = datetime(2024, 12, 31)
MAX_CHECKOUT_DAYS = 21  # Maximum checkout period in days
GRADE_LEVELS = [9, 10, 11, 12]
STUDENT_INTERESTS = ['Science', 'Technology', 'Arts', 'Literature', 'History', 
                     'Sports', 'Mathematics', 'Social Studies', 'Foreign Languages']

# Create datasets directory if it doesn't exist
os.makedirs('datasets', exist_ok=True)

# List of popular genres in high school libraries
genres = [
    'Young Adult Fiction', 'Science Fiction', 'Fantasy', 'Mystery', 'Historical Fiction',
    'Romance', 'Thriller', 'Biography', 'Memoir', 'Non-fiction', 'Science', 'Poetry',
    'Graphic Novel', 'Classic Literature', 'Adventure', 'Dystopian', 'Contemporary',
    'Horror', 'Humor', 'Drama'
]

# Topics that might interest high school students
topics = [
    'Coming of Age', 'Identity', 'Friendship', 'Family', 'School', 'Love', 'Loss',
    'Mental Health', 'Social Issues', 'Technology', 'Science', 'History', 'Sports',
    'Music', 'Art', 'Environment', 'Politics', 'Culture', 'Diversity', 'LGBTQ+',
    'Survival', 'Mystery', 'Adventure', 'Fantasy', 'Sci-Fi', 'Dystopia', 'Utopia'
]

# Age categories for Young Adult books
age_categories = ['Middle Grade (10-13)', 'Young Teen (12-14)', 'Teen (14-17)', 'Young Adult (15-18)', 'New Adult (16-22)']

# Popular authors for Young Adult and educational books
authors = [
    'John Green', 'J.K. Rowling', 'Suzanne Collins', 'Angie Thomas', 'Tomi Adeyemi',
    'Cassandra Clare', 'Sarah J. Maas', 'Veronica Roth', 'Marie Lu', 'Neal Shusterman',
    'Markus Zusak', 'Rainbow Rowell', 'Leigh Bardugo', 'Jason Reynolds', 'Sabaa Tahir',
    'Rick Riordan', 'Becky Albertalli', 'Adam Silvera', 'Nicola Yoon', 'Tahereh Mafi',
    'Jenny Han', 'Nic Stone', 'Stephen Hawking', 'Neil deGrasse Tyson', 'Yuval Noah Harari',
    'Malcolm Gladwell', 'Steven Levitt', 'Stephen Dubner', 'Bill Nye', 'Brian Greene',
    'Walter Isaacson', 'Jane Goodall', 'Hope Jahren', 'Temple Grandin', 'Mary Roach'
]

# Publishers specializing in Young Adult and educational content
publishers = [
    'Penguin Random House', 'HarperCollins', 'Simon & Schuster', 'Macmillan Publishers',
    'Scholastic', 'Hachette Book Group', 'Little, Brown and Company', 'Bloomsbury',
    'Disney-Hyperion', 'Random House Children\'s Books', 'HarperTeen', 'Simon Pulse',
    'Candlewick Press', 'First Second Books', 'National Geographic Learning',
    'W. W. Norton & Company', 'Oxford University Press', 'Cambridge University Press',
    'Pearson Education', 'McGraw Hill', 'Dover Publications', 'DK Publishing'
]

def generate_random_date(start_date, end_date):
    """Generate a random date between start_date and end_date"""
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_days = np.random.randint(0, days_between_dates)
    return start_date + timedelta(days=random_days)

def generate_book_data():
    """Generate dataset of books with metadata"""
    data = []
    
    for i in tqdm(range(NUM_BOOKS), desc="Generating books"):
        # Basic book information
        book_id = f'BK{i+1:05d}'
        title = f'Sample Book Title {i+1}'  # In a real dataset, you would have actual titles
        author = np.random.choice(authors)
        genre = np.random.choice(genres)
        publisher = np.random.choice(publishers)
        publication_year = np.random.randint(1990, 2024)
        
        # Book characteristics
        page_count = np.random.randint(100, 800)
        is_series = np.random.choice([True, False], p=[0.4, 0.6])
        series_position = np.random.randint(1, 8) if is_series else None
        reading_level = np.random.choice(['Easy', 'Intermediate', 'Advanced'])
        age_category = np.random.choice(age_categories)
        has_movie_adaptation = np.random.choice([True, False], p=[0.2, 0.8])
        
        # Content and appeal factors
        main_topics = np.random.choice(topics, size=np.random.randint(1, 5), replace=False).tolist()
        has_diverse_characters = np.random.choice([True, False], p=[0.4, 0.6])
        awards_count = np.random.choice([0, 0, 0, 0, 1, 1, 2, 3], p=[0.6, 0.1, 0.1, 0.05, 0.05, 0.05, 0.03, 0.02])
        average_goodreads_rating = np.random.uniform(2.5, 4.9)  # Range typically found on Goodreads
        
        # Availability and format
        has_ebook = np.random.choice([True, False], p=[0.7, 0.3])
        has_audiobook = np.random.choice([True, False], p=[0.5, 0.5])
        copies_available = np.random.randint(1, 6)
        
        # Generate acquisition date (when the library got the book)
        acquisition_date = generate_random_date(START_DATE, END_DATE - timedelta(days=180))
        acquisition_date_str = acquisition_date.strftime('%Y-%m-%d')
        
        # School curriculum and recommendation factors
        is_required_reading = np.random.choice([True, False], p=[0.1, 0.9])
        is_teacher_recommended = np.random.choice([True, False], p=[0.3, 0.7])
        curriculum_subjects = []
        if is_required_reading or is_teacher_recommended:
            curriculum_subjects = np.random.choice(
                ['English', 'History', 'Science', 'Social Studies', 'Art', 'Foreign Language'],
                size=np.random.randint(1, 3),
                replace=False
            ).tolist()
        
        # Book data record
        book_data = {
            'book_id': book_id,
            'title': title,
            'author': author,
            'genre': genre,
            'publisher': publisher,
            'publication_year': publication_year,
            'page_count': page_count,
            'is_series': is_series,
            'series_position': series_position,
            'reading_level': reading_level,
            'age_category': age_category,
            'has_movie_adaptation': has_movie_adaptation,
            'main_topics': main_topics,
            'has_diverse_characters': has_diverse_characters,
            'awards_count': awards_count,
            'average_goodreads_rating': round(average_goodreads_rating, 2),
            'has_ebook': has_ebook,
            'has_audiobook': has_audiobook,
            'copies_available': copies_available,
            'acquisition_date': acquisition_date_str,
            'is_required_reading': is_required_reading,
            'is_teacher_recommended': is_teacher_recommended,
            'curriculum_subjects': curriculum_subjects
        }
        
        data.append(book_data)
    
    return pd.DataFrame(data)

def generate_student_data(num_students=800):
    """Generate a dataset of fictional high school students"""
    students = []
    student_ids = np.arange(1, num_students + 1)
    
    for student_id in tqdm(student_ids, desc="Generating students"):
        grade = np.random.choice(GRADE_LEVELS)
        age = grade + 4 + np.random.choice([-1, 0, 0, 0, 1], p=[0.05, 0.25, 0.4, 0.25, 0.05])  # Most students are at typical age for grade
        gender = np.random.choice(['Male', 'Female', 'Non-binary'], p=[0.48, 0.48, 0.04])
        gpa = np.random.normal(3.0, 0.7)  # Normal distribution with mean 3.0 and std 0.7
        gpa = max(0.0, min(4.0, gpa))  # Clip to valid GPA range
        
        # Assign primary interests, weighted by academic performance
        if gpa > 3.5:
            interest_weights = [0.2, 0.2, 0.15, 0.15, 0.1, 0.05, 0.1, 0.03, 0.02]  # Higher chance of academic interests
        elif gpa > 2.5:
            interest_weights = [0.15, 0.15, 0.15, 0.15, 0.1, 0.1, 0.05, 0.1, 0.05]  # Balanced interests
        else:
            interest_weights = [0.05, 0.05, 0.2, 0.15, 0.05, 0.3, 0.05, 0.1, 0.05]  # Higher chance of non-academic interests
        
        primary_interests = np.random.choice(
            STUDENT_INTERESTS, 
            size=np.random.randint(1, 4), 
            replace=False, 
            p=interest_weights
        ).tolist()
        
        # Participation in school activities based on interests and academic performance
        activities = []
        if 'Science' in primary_interests or 'Technology' in primary_interests:
            if np.random.random() < 0.7:
                activities.append('Science Club')
            if np.random.random() < 0.5:
                activities.append('Robotics Team')
        
        if 'Arts' in primary_interests:
            if np.random.random() < 0.8:
                activities.append('Art Club')
            if np.random.random() < 0.5:
                activities.append('Theater')
                
        if 'Literature' in primary_interests:
            if np.random.random() < 0.8:
                activities.append('Book Club')
            if np.random.random() < 0.5:
                activities.append('School Newspaper')
                
        if 'Sports' in primary_interests:
            if np.random.random() < 0.9:
                activities.append(np.random.choice(['Basketball Team', 'Soccer Team', 'Track Team', 'Swimming']))
        
        # Reading frequency correlates with certain interests and GPA
        reading_habit_factor = gpa / 4.0  # Base factor from GPA
        if 'Literature' in primary_interests:
            reading_habit_factor += 0.3
        if 'History' in primary_interests:
            reading_habit_factor += 0.2
        
        reading_habit_factor = min(1.0, reading_habit_factor)
        
        if reading_habit_factor > 0.8:
            reading_frequency = 'Frequent'
        elif reading_habit_factor > 0.5:
            reading_frequency = 'Regular'
        elif reading_habit_factor > 0.3:
            reading_frequency = 'Occasional'
        else:
            reading_frequency = 'Rare'
        
        # Reading preferences based on interests
        preferred_genres = []
        if 'Science' in primary_interests or 'Technology' in primary_interests:
            preferred_genres.extend(['Science Fiction', 'Non-fiction'])
        if 'Arts' in primary_interests:
            preferred_genres.extend(['Graphic Novel', 'Poetry'])
        if 'Literature' in primary_interests:
            preferred_genres.extend(['Classic Literature', 'Young Adult Fiction'])
        if 'History' in primary_interests:
            preferred_genres.extend(['Historical Fiction', 'Biography'])
        if 'Sports' in primary_interests:
            preferred_genres.extend(['Biography', 'Non-fiction'])
            
        if not preferred_genres:  # If no specific genres assigned based on interests
            preferred_genres = np.random.choice(genres, size=np.random.randint(1, 4), replace=False).tolist()
        
        # Ensure no duplicates in the preferred genres
        preferred_genres = list(set(preferred_genres))
        
        student = {
            'student_id': f'ST{student_id:04d}',
            'grade': grade,
            'age': age,
            'gender': gender,
            'gpa': round(gpa, 2),
            'primary_interests': primary_interests,
            'activities': activities,
            'reading_frequency': reading_frequency,
            'preferred_genres': preferred_genres
        }
        
        students.append(student)
    
    return pd.DataFrame(students)

def generate_checkout_history(books_df, students_df):
    """Generate checkout history data for books and students"""
    checkouts = []
    
    # Get all book and student IDs
    book_ids = books_df['book_id'].tolist()
    student_ids = students_df['student_id'].tolist()
    
    # Define functions for book popularity factors
    def calculate_base_popularity(book):
        """Calculate a base popularity score for a book based on its attributes"""
        popularity = 0
        
        # More recent books tend to be more popular
        recency_factor = min(1.0, (book['publication_year'] - 1990) / 30)  # Scale from 0 to 1
        popularity += recency_factor * 3
        
        # Goodreads ratings influence popularity
        rating_factor = (book['average_goodreads_rating'] - 2.5) / 2.5  # Scale from 0 to 1
        popularity += rating_factor * 2
        
        # Books with movie adaptations tend to be more popular
        if book['has_movie_adaptation']:
            popularity += 1.5
            
        # Books that are part of a series often have higher readership
        if book['is_series']:
            popularity += 0.5
            
        # Required reading or teacher recommended books get borrowed more
        if book['is_required_reading']:
            popularity += 2
        if book['is_teacher_recommended']:
            popularity += 1
            
        # Award-winning books may be more popular
        popularity += book['awards_count'] * 0.5
        
        # Add some randomness (some books just become popular for reasons not captured in the data)
        popularity += np.random.normal(0, 1)  # Random factor with standard deviation of 1
        
        # Normalize to a probability-like scale
        return max(0.1, min(10, popularity))
    
    print("Calculating base popularity for books...")
    # Calculate base popularity for each book
    books_df['base_popularity'] = books_df.apply(calculate_base_popularity, axis=1)
    
    # Generate multiple checkout events
    current_date = END_DATE
    
    # Students in different grades check out different numbers of books on average
    checkout_count_by_grade = {9: 20, 10: 25, 11: 30, 12: 35}
    
    # Process each student
    total_checkouts = 0
    student_count = len(students_df)
    
    for i, (_, student) in enumerate(tqdm(list(students_df.iterrows()), desc="Generating checkout history")):
        student_id = student['student_id']
        grade = student['grade']
        preferred_genres = student['preferred_genres']
        reading_frequency = student['reading_frequency']
        
        # Determine how many books this student checks out based on reading frequency
        frequency_multiplier = {
            'Frequent': 1.5,
            'Regular': 1.0,
            'Occasional': 0.6,
            'Rare': 0.3
        }[reading_frequency]
        
        # Calculate the baseline number of checkouts for this student
        base_checkout_count = checkout_count_by_grade[grade] * frequency_multiplier
        
        # Add some randomness to the checkout count
        target_checkout_count = int(np.random.normal(base_checkout_count, base_checkout_count * 0.2))
        target_checkout_count = max(1, target_checkout_count)  # At least 1 checkout
        total_checkouts += target_checkout_count
        
        # Generate checkouts for this student
        # Book preferences: students are more likely to check out books that match their interests/preferences
        # For each genre the student prefers, boost books in that genre
        for _ in range(target_checkout_count):
            # Copy the base popularity scores
            adjusted_popularity = books_df['base_popularity'].copy()
            
            # Boost books in preferred genres
            for genre in preferred_genres:
                genre_mask = books_df['genre'] == genre
                adjusted_popularity[genre_mask] *= 2.0
                
            # Boost books with topics related to the student's interests
            for interest in student['primary_interests']:
                # Find books with related topics
                for idx, book in books_df.iterrows():
                    topics = book['main_topics']
                    if interest in topics or any(t for t in topics if interest.lower() in t.lower()):
                        adjusted_popularity[idx] *= 1.5
            
            # Required reading is more likely for relevant grade levels
            required_mask = books_df['is_required_reading']
            if grade >= 11:  # Upper classmen are more likely to read classics & required books
                adjusted_popularity[required_mask] *= 1.5
            
            # Probability of selecting each book
            selection_probs = adjusted_popularity / adjusted_popularity.sum()
            
            # Select a book based on the adjusted popularity
            selected_book_idx = np.random.choice(range(len(books_df)), p=selection_probs)
            selected_book = books_df.iloc[selected_book_idx]
            
            # Generate checkout data
            book_id = selected_book['book_id']
            checkout_date = generate_random_date(START_DATE, END_DATE - timedelta(days=30))
            
            # Duration relates to:
            # - Book length (more pages = longer checkout)
            # - Student's reading frequency (frequent readers may finish faster)
            # - Popularity (popular books might be read quickly or kept longer due to demand)
            base_days = min(21, max(7, int(selected_book['page_count'] / 50)))  # Base on page count
            
            # Adjust for reading frequency
            frequency_checkout_multiplier = {
                'Frequent': 0.8,  # Finish faster
                'Regular': 1.0,
                'Occasional': 1.2,
                'Rare': 1.5      # Take longer
            }[reading_frequency]
            
            checkout_days = int(base_days * frequency_checkout_multiplier)
            checkout_days = min(MAX_CHECKOUT_DAYS, max(3, checkout_days))  # Between 3 and MAX_CHECKOUT_DAYS
            
            # Add some randomness to the checkout duration
            checkout_days = max(1, int(np.random.normal(checkout_days, checkout_days * 0.2)))
            
            return_date = checkout_date + timedelta(days=checkout_days)
            
            # Record the checkout
            checkout = {
                'book_id': book_id,
                'student_id': student_id,
                'checkout_date': checkout_date.strftime('%Y-%m-%d'),
                'return_date': return_date.strftime('%Y-%m-%d'),
                'checkout_duration_days': checkout_days
            }
            
            checkouts.append(checkout)
    
    return pd.DataFrame(checkouts)

def calculate_popularity_metrics(books_df, checkouts_df):
    """Calculate various book popularity metrics based on checkout history"""
    print("Calculating popularity metrics...")
    
    # Total checkouts per book
    book_checkout_counts = checkouts_df['book_id'].value_counts().reset_index()
    book_checkout_counts.columns = ['book_id', 'total_checkouts']
    
    # Average checkout duration per book
    avg_duration = checkouts_df.groupby('book_id')['checkout_duration_days'].mean().reset_index()
    avg_duration.columns = ['book_id', 'avg_checkout_duration']
    
    # Calculate recency factor - more weight to recent checkouts
    checkouts_df['checkout_date'] = pd.to_datetime(checkouts_df['checkout_date'])
    max_date = checkouts_df['checkout_date'].max()
    
    # Calculate days since the end date for each checkout
    checkouts_df['days_since_checkout'] = (max_date - checkouts_df['checkout_date']).dt.days
    
    # Calculate a recency weighted score
    checkouts_df['recency_weight'] = 1 / (1 + checkouts_df['days_since_checkout'] / 180)  # Half a year decay
    
    # Calculate the recency weighted score for each book
    recency_score = checkouts_df.groupby('book_id')['recency_weight'].sum().reset_index()
    recency_score.columns = ['book_id', 'recency_score']
    
    # Merge popularity metrics with the books dataframe
    popularity_metrics = pd.merge(book_checkout_counts, avg_duration, on='book_id', how='outer')
    popularity_metrics = pd.merge(popularity_metrics, recency_score, on='book_id', how='outer')
    
    # Fill any NaN values (books with no checkouts)
    popularity_metrics = popularity_metrics.fillna(0)
    
    # Create a popularity score that combines these metrics
    # The formula weights recent checkouts more heavily
    popularity_metrics['popularity_score'] = (
        popularity_metrics['total_checkouts'] * 0.5 +
        popularity_metrics['recency_score'] * 0.5
    )

    # Ensure popularity_score is filled with zeros for books with no checkouts
    popularity_metrics['popularity_score'] = popularity_metrics['popularity_score'].fillna(0)

    # Normalize the popularity score to a 0-100 scale
    max_score = popularity_metrics['popularity_score'].max()
    if max_score > 0:  # Avoid division by zero
        popularity_metrics['popularity_score'] = (popularity_metrics['popularity_score'] / max_score) * 100
    
    # Calculate checkouts per copy
    books_with_metrics = pd.merge(books_df, popularity_metrics, on='book_id', how='left')
    books_with_metrics['checkouts_per_copy'] = books_with_metrics['total_checkouts'] / books_with_metrics['copies_available']
    
    # Normalize checkouts per copy to a 0-100 scale
    max_checkouts_per_copy = books_with_metrics['checkouts_per_copy'].max()
    if max_checkouts_per_copy > 0:  # Avoid division by zero
        books_with_metrics['checkouts_per_copy_score'] = (books_with_metrics['checkouts_per_copy'] / max_checkouts_per_copy) * 100
    else:
        books_with_metrics['checkouts_per_copy_score'] = 0
    
    return books_with_metrics

def main():
    print("Generating book data...")
    books_df = generate_book_data()
    
    print("Generating student data...")
    students_df = generate_student_data()
    
    checkouts_df = generate_checkout_history(books_df, students_df)
    
    books_with_metrics = calculate_popularity_metrics(books_df, checkouts_df)
    
    # Save the generated data
    print("Saving datasets...")
    books_df.to_csv('datasets/books.csv', index=False)
    students_df.to_csv('datasets/students.csv', index=False)
    checkouts_df.to_csv('datasets/checkouts.csv', index=False)
    books_with_metrics.to_csv('datasets/books_with_metrics.csv', index=False)
    
    # Print summary statistics
    print(f"\nGenerated {len(books_df)} books")
    print(f"Generated {len(students_df)} students")
    print(f"Generated {len(checkouts_df)} checkout records")
    
    # Print sample of the data with the highest popularity scores
    top_books = books_with_metrics.sort_values('popularity_score', ascending=False).head(10)[['book_id', 'title', 'genre', 'publication_year', 'total_checkouts', 'popularity_score']]
    print("\nTop 10 Most Popular Books:")
    print(top_books)

if __name__ == "__main__":
    main()
