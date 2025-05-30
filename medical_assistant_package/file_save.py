from datetime import datetime
import numpy as np
import os
import json

def save_to_json(record, folder="Medical_record"):
    
        if not os.path.exists(folder):
            os.makedirs(folder)

   
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{folder}/{record['name']}_{timestamp}.json"
        
        for key, value in record.items():
            if isinstance(value, np.ndarray):
                record[key] = value.tolist()
    
        with open(filename, "w") as f:
            json.dump(record, f, indent=4)

        return filename 