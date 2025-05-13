#!/usr/bin/env hy

;; Import required modules with proper Hy syntax
(import datetime)
(import json) 
(import os)
(require hyrule [-> ->> as-> case]) ; Import useful macros from hyrule

;; Define the file for storing todos
(setv todo-file "todos.json")

;; Task structure functions
(defn create-task [description]
  "Creates a new task dictionary with description, timestamp and completion status"
  {"description" description
   "created" (str (datetime.datetime.now))
   "completed" False})

;; File operations
(defn load-tasks []
  "Loads tasks from JSON file, returns empty list if file doesn't exist"
  (if (os.path.exists todo-file)
    (with [f (open todo-file "r")]
      (json.loads (.read f)))
    []))

(defn save-tasks [tasks]
  "Saves tasks list to JSON file with pretty printing"
  (with [f (open todo-file "w")]
    (.write f (json.dumps tasks :indent 2))))

;; Task operations
(defn add-task [tasks description]
  "Adds a new task to the tasks list"
  (+ tasks [(create-task description)]))

(defn complete-task [tasks index]
  "Marks task at given index as complete, returns updated tasks list"
  (if (and (>= index 0) (< index (len tasks)))
    (do
      ;; Use the thread-first macro to make the code more readable
      (-> (get tasks index)
          (as-> task (do
                       (setv (get task "completed") True)
                       task)))
      tasks)
    (do
      (print "Invalid task index")
      tasks)))

(defn list-tasks [tasks]
  "Displays all tasks with their completion status"
  (if (= (len tasks) 0)
    (print "No tasks found.")
    (do
      (print "\nYour To-Do List:")
      (print "----------------")
      (for [[i task] (enumerate tasks)]
        (print (format "[{}] {} {}" 
                       (str i)
                       (if (get task "completed") "[X]" "[ ]")
                       (get task "description")))))))

;; Main function
(defn main []
  "Main program loop handling user commands"
  (setv tasks (load-tasks))
  (while True
    (print "\nCommands: add, complete, list, quit")
    (setv command (input "Enter command: "))
    
    (case command
      ;; Add new task
      "add" (do
              (setv desc (input "Task description: "))
              (->> desc
                   (add-task tasks)
                   (setv tasks))
              (save-tasks tasks)
              (print "Task added."))
      
      ;; Complete existing task
      "complete" (do
                   (try
                     (do
                       (setv index (int (input "Task index: ")))
                       (setv tasks (complete-task tasks index))
                       (save-tasks tasks)
                       (print "Task marked as complete."))
                     (except [ValueError]
                       (print "Please enter a valid number."))))
      
      ;; List all tasks
      "list" (list-tasks tasks)
      
      ;; Exit program
      "quit" (do
               (print "Goodbye!")
               (break))
      
      ;; Handle unknown commands
      _ (print "Unknown command."))))

;; Execute the main function when the script is run directly
(when (= __name__ "__main__")
  (main))