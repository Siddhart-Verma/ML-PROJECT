'''from pymongo import MongoClient

# Function to clean the dataset by removing specified fields
def clean_dataset():
    # MongoDB connection
    client = MongoClient('mongodb://localhost:27017/')  # Update with your connection string if needed
    db = client['project']
    collection = db['future']

    # Fields to remove from the dataset
    fields_to_remove = ['address', 'Pstatus', 'reason', 'nursery', 'romantic', 'famrel', 'Dalc', 'Walc']

    # Removing fields from all documents
    update_query = {'$unset': {field: "" for field in fields_to_remove}}
    result = collection.update_many({}, update_query)

    print(f"Fields removed from {result.modified_count} documents.")

if __name__ == "__main__":
    clean_dataset()'''
    
    
'''from pymongo import MongoClient

# Function to clean the dataset by removing specified fields and duplicates
def clean_dataset():
    # MongoDB connection
    client = MongoClient('mongodb://localhost:27017/')  # Update with your connection string if needed
    db = client['project']
    collection = db['future']

    # Fields to remove from the dataset
    fields_to_remove = ['address', 'Pstatus', 'reason', 'nursery', 'romantic', 'famrel', 'Dalc', 'Walc']

    # Removing fields from all documents
    update_query = {'$unset': {field: "" for field in fields_to_remove}}
    result = collection.update_many({}, update_query)
    print(f"Fields removed from {result.modified_count} documents.")

    # Removing duplicate documents based on unique fields
    unique_field = 'unique_identifier'  # Replace with the actual unique field
    pipeline = [
        {"$group": {
            "_id": f"${unique_field}",
            "ids": {"$push": "$_id"},
            "count": {"$sum": 1}
        }},
        {"$match": {"count": {"$gt": 1}}}
    ]

    duplicates = collection.aggregate(pipeline)
    for doc in duplicates:
        ids_to_delete = doc['ids'][1:]  # Keep one document, remove the rest
        collection.delete_many({"_id": {"$in": ids_to_delete}})
    print("Duplicate documents removed.")

if __name__ == "__main__":
    clean_dataset()'''



from pymongo import MongoClient

# Function to clean the dataset by removing specified fields
def clean_dataset():
    # MongoDB connection
    client = MongoClient('mongodb://localhost:27017/')  # Update with your connection string if needed
    db = client['project']
    collection = db['future']

    # Fields to remove from the dataset
    fields_to_remove = ['address', 'Pstatus', 'reason', 'nursery', 'romantic', 'famrel', 'Dalc', 'Walc','guardian', 'famsize', 'Medu',
                        'Fedu', 'guardian', 'traveltime', 'paid', 'goout', 'health','school','schoolsup','famsup']

    # Removing fields from all documents
    update_query = {'$unset': {field: "" for field in fields_to_remove}}
    result = collection.update_many({}, update_query)
    print(f"Fields removed from {result.modified_count} documents.")

    # Removing duplicate documents (if any exist despite _id being unique)
    pipeline = [
        {"$group": {
            "_id": "$_id",  # Group by the unique _id field
            "count": {"$sum": 1},
            "ids": {"$push": "$_id"}
        }},
        {"$match": {"count": {"$gt": 1}}}  # Find groups where count > 1
    ]

    duplicates = collection.aggregate(pipeline)
    for doc in duplicates:
        ids_to_delete = doc['ids'][1:]  # Keep one document, remove the rest
        collection.delete_many({"_id": {"$in": ids_to_delete}})
    print("Duplicate documents removed, if any.")

if __name__ == "__main__":
    clean_dataset()

