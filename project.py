import sqlite3
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import date,datetime
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import csv
import threading

# Directory for storing face images
FACE_DATA_DIR = "faces"

# Global flag to prevent multiple recognitions
recognition_in_progress = False


# Database Initialization
def initialize_db():
    connection = sqlite3.connect("attendance.db")
    cursor = connection.cursor()

    # Drop the existing tables if they exist
    #cursor.execute("DROP TABLE IF EXISTS students")
    #cursor.execute("DROP TABLE IF EXISTS attendance")

    # Recreate the tables with the correct schema
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        enrollment_no TEXT UNIQUE NOT NULL,
        class_roll_no TEXT UNIQUE NOT NULL,
        course TEXT NOT NULL,
        semester TEXT NOT NULL
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        enrollment_no TEXT,
        name TEXT NOT NULL,
        date DATE NOT NULL,
        time TIME NOT NULL,
        status TEXT CHECK(status IN ('Present', 'Absent')) NOT NULL,
        FOREIGN KEY (enrollment_no) REFERENCES students(enrollment_no)
    )
    """)

    connection.commit()
    connection.close()

# Add a new student
def add_student(name, enrollment_no, class_roll_no, course, semester):
    connection = sqlite3.connect("attendance.db")
    cursor = connection.cursor()
    try:
        cursor.execute("INSERT INTO students (name, enrollment_no, class_roll_no, course, semester) VALUES (?, ?, ?, ?, ?)", 
                       (name, enrollment_no, class_roll_no, course, semester))
        connection.commit()
        capture_faces(class_roll_no, name)  # Capture face data for the student
        messagebox.showinfo("Success", "Student added successfully!")
    except sqlite3.IntegrityError:
        messagebox.showerror("Error", "Roll Number already exists!")
    finally:
        connection.close()

# Capture and save face data
def capture_faces( class_roll_no, name):
    if not os.path.exists(FACE_DATA_DIR):
        os.makedirs(FACE_DATA_DIR)

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_count += 1
            file_path = os.path.join(FACE_DATA_DIR, f"{class_roll_no}_{face_count}.jpg")
            cv2.imwrite(file_path, face)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Use OpenCV's imshow instead of matplotlib
        cv2.imshow("Capture Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break  

        if face_count >= 10:  # Stop capturing after 10 faces
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {face_count} faces for {name}.")

# Train Face Recognizer
def train_recognizer():
    face_data = []
    labels = []

    # Load face images and labels from the directory
    for file_name in os.listdir(FACE_DATA_DIR):
        if file_name.endswith(".jpg"):
            # Read the image in grayscale
            img = cv2.imread(os.path.join(FACE_DATA_DIR, file_name), cv2.IMREAD_GRAYSCALE)

            # Resize the image to a fixed size (e.g., 100x100)
            img_resized = cv2.resize(img, (100, 100))  # Resize to 100x100 pixels

            # Flatten the resized image and append to the list
            face_data.append(img_resized.flatten())  # Flatten the image
            labels.append(file_name.split("_")[0])  # Assuming roll number is part of the filename

    # Check if we have enough samples for training
    if len(face_data) < 1:
        print("No face data available for training.")
        return None

    # Set n_neighbors dynamically based on the training data size
    n_neighbors = min(3, len(face_data)) 
    
    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(face_data, labels)
    return knn


# Recognize Faces and Mark Attendance
def recognize_faces(knn):
    if knn is None:
        print("Face recognizer not trained. Cannot proceed with recognition.")
        return

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    face_count=0
    # Store a list of students for whom attendance has been marked
    marked_students_today = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access the camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        for (x, y, w, h) in faces:
            face_count+=1
            face = cv2.resize(gray[y:y + h, x:x + w], (100, 100)).flatten()
            try:
                # Predict the class roll number
                class_roll_no = knn.predict([face])[0]

                # Connect to the database
                connection = sqlite3.connect("attendance.db")
                cursor = connection.cursor()

                # Fetch the student's enrollment number and name using the roll number
                cursor.execute("SELECT enrollment_no, name FROM students WHERE class_roll_no = ?", (class_roll_no,))
                result = cursor.fetchone()

                if result:
                    enrollment_no, name = result

                    # Skip processing if attendance is already marked for this student today
                    if enrollment_no in marked_students_today:
                        print(f"Attendance already marked for {name} ({class_roll_no}) today.")
                        continue  # Skip to the next face without marking attendance again

                    # Check if attendance has already been marked for this student today
                    cursor.execute("""
                        SELECT id FROM attendance WHERE enrollment_no = ? AND date = ?
                    """, (enrollment_no, date.today()))
                    existing_record = cursor.fetchone()

                    if existing_record:
                        print(f"Attendance already marked for {name} ({class_roll_no}) today.")
                        messagebox.showinfo("Attendance Marked", f"Attendance already recorded for {name} ({class_roll_no}) today.")
                        # Skip this student from further processing and add to the marked set
                        marked_students_today.add(enrollment_no)
                    else:
                        # Mark the student as present
                        cursor.execute("""
                            INSERT INTO attendance (enrollment_no, name, date, time, status)
                            VALUES (?, ?, ?, ?, ?)
                        """, (enrollment_no, name, date.today(), datetime.now().strftime('%H:%M:%S'), "Present"))
                        connection.commit()
                        print(f"Attendance marked for Roll: {class_roll_no} (Name: {name})")
                        messagebox.showinfo("Success", f"Attendance marked for {name} ({class_roll_no}).")
                        # Add the student to the marked list
                        marked_students_today.add(enrollment_no)

                connection.close()

                # Draw the bounding box and display the roll number
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Roll: {class_roll_no}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error during recognition: {e}")
            if face_count>=1:
                break

        # Display the frame
        cv2.imshow("Face Recognition", frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Exiting face recognition...")
            break
        if face_count>=1:
                break

    cap.release()
    cv2.destroyAllWindows()


# Fetch attendance records
def fetch_attendance():
    connection = sqlite3.connect("attendance.db")
    cursor = connection.cursor()
    cursor.execute("""
    SELECT s.name, s.class_roll_no, a.date, a.status
    FROM students s
    JOIN attendance a ON s.enrollment_no = a.enrollment_no
    ORDER BY a.date DESC

    """)
    records = cursor.fetchall()
    connection.close()

    # Log the records to check
    print("Fetched Attendance Records:", records)
    return records

# Export attendance to CSV
def export_to_csv():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance")
    records = cursor.fetchall()

    with open('attendance.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['S.NO', 'ID','Student Name', 'Date','Time','Status'])
        writer.writerows(records)

    messagebox.showinfo("Info", "Attendance exported to attendance.csv.")
    conn.close()


def check_database():
    connection = sqlite3.connect("attendance.db")
    cursor = connection.cursor()

    # Check all students
    cursor.execute("SELECT * FROM students")
    students = cursor.fetchall()
    print("Students:", students)

    # Check all attendance records
    cursor.execute("SELECT * FROM attendance")
    attendance = cursor.fetchall()
    print("Attendance:", attendance)

    connection.close()


# Start recognition in a separate thread
def start_recognition():
    knn = train_recognizer()
    if knn:
        # Start a new thread to run recognize_faces without blocking the main UI thread
        recognition_thread = threading.Thread(target=recognize_faces, args=(knn,))
        recognition_thread.start()
    else:
        messagebox.showerror("Error", "Face recognizer training failed.")


# Main GUI Application
def main():
    initialize_db()

    root = tk.Tk()
    root.title("Attendance Management System")
    root.geometry("800x600")

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Add Student Tab
    add_student_frame = ttk.Frame(notebook)
    notebook.add(add_student_frame, text="Add Student")

    tk.Label(add_student_frame, text="Name:").grid(row=0, column=0, padx=10, pady=10)
    tk.Label(add_student_frame, text="Enrollment Number:").grid(row=1, column=0, padx=10, pady=10)
    tk.Label(add_student_frame, text="Class Roll Number:").grid(row=2, column=0, padx=10, pady=10)
    tk.Label(add_student_frame, text="Course:").grid(row=3, column=0, padx=10, pady=10)
    tk.Label(add_student_frame, text="Semester:").grid(row=4, column=0, padx=10, pady=10)

    name_entry = tk.Entry(add_student_frame)
    enrollment_no_entry = tk.Entry(add_student_frame)
    class_roll_no_entry = tk.Entry(add_student_frame)
    course_entry = tk.Entry(add_student_frame)
    semester_entry = tk.Entry(add_student_frame)

    name_entry.grid(row=0, column=1, padx=10, pady=10)
    enrollment_no_entry.grid(row=1, column=1, padx=10, pady=10)
    class_roll_no_entry.grid(row=2, column=1, padx=10, pady=10)
    course_entry.grid(row=3, column=1, padx=10, pady=10)
    semester_entry.grid(row=4, column=1, padx=10, pady=10)

    def add_student_callback():
        name = name_entry.get()
        enrollment_no = enrollment_no_entry.get()
        class_roll_no = class_roll_no_entry.get()
        course = course_entry.get()
        semester = semester_entry.get()
        if name and enrollment_no and class_roll_no and course and semester:
            add_student(name, enrollment_no, class_roll_no, course, semester)
        else:
            messagebox.showerror("Error", "All fields are required!")

    tk.Button(add_student_frame, text="Add Student", command=add_student_callback).grid(row=6, column=0, columnspan=2, pady=10)

    # Recognize Attendance Tab
    recognize_frame = ttk.Frame(notebook)
    notebook.add(recognize_frame, text="Face Recognition Attendance")

    start_button = tk.Button(recognize_frame, text="Start Recognition", command=start_recognition)
    start_button.pack(pady=20)

    check_database()
    
    # View Attendance Tab
    view_attendance_frame = ttk.Frame(notebook)
    notebook.add(view_attendance_frame, text="View Attendance")

    tree = ttk.Treeview(view_attendance_frame, columns=("Name", "Class Roll No", "Date", "Status"), show="headings")
    tree.heading("Name", text="Name")
    tree.heading("Class Roll No", text="Class Roll No")
    tree.heading("Date", text="Date")
    tree.heading("Status", text="Status")
    tree.pack(fill=tk.BOTH, expand=True)

    def load_attendance():
        for row in tree.get_children():
           tree.delete(row)
        records = fetch_attendance()
        for record in records:
            tree.insert("", "end", values=record)

    tk.Button(view_attendance_frame, text="Refresh", command=load_attendance).pack(pady=10)
    tk.Button(view_attendance_frame, text="Export to CSV", command=export_to_csv).pack(pady=10)
    root.mainloop()
if __name__ == "__main__":
    main()

