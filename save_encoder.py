import pickle
import os

STATE_MAPPING = {
    0: 'Dadra & Nagar Haveli', 1: 'Daman & Diu', 2: 'Delhi', 3: 'Goa', 
    4: 'Gujarat', 5: 'Madhya Pradesh', 6: 'Odisha', 7: 'Tamil Nadu', 
    8: 'Telangana', 9: 'West Bengal'
}

TYPE_MAPPING = {
    0: 'Industrial Area', 1: 'RIRUO', 2: 'Residential, Rural and other Areas'
}

LOC_MAPPING = {
    0: 'ANKLESHWAR', 1: 'Ahmedabad', 2: 'Amona', 3: 'Angul', 4: 'Ankleshwar', 
    5: 'Assanora', 6: 'Balasore', 7: 'Barrackpore', 8: 'Berhampur', 9: 'Bharuch', 
    10: 'Bhopal', 11: 'Bhubaneswar', 12: 'Bhuj', 13: 'Bicholim', 14: 'Chennai', 
    15: 'Codli', 16: 'Coimbatore', 17: 'Cuddalore', 18: 'Cuncolim', 19: 'Curchorem', 
    20: 'Cuttack', 21: 'Daman', 22: 'Delhi', 23: 'Durgapur', 24: 'Gwalior', 
    25: 'Honda', 26: 'Howrah', 27: 'Jabalpur', 28: 'Jamnagar', 29: 'Kalinga Nagar', 
    30: 'Keonjhar', 31: 'Khadoli', 32: 'Kolkata', 33: 'Konark', 34: 'Kundaim', 
    35: 'MORBI', 36: 'Madurai', 37: 'Mapusa', 38: 'Margao', 39: 'Mettur', 
    40: 'Nagda', 41: 'Paradeep', 42: 'Ponda', 43: 'Puri', 44: 'Rajkot', 
    45: 'Rayagada', 46: 'Rourkela', 47: 'SANAND', 48: 'Sagar', 49: 'Salem', 
    50: 'Sambalpur', 51: 'Sangareddy', 52: 'Sanguem', 53: 'Sarigam', 54: 'Singrauli', 
    55: 'Surat', 56: 'Talcher', 57: 'Thoothukudi', 58: 'Tilamol', 59: 'Trichy', 
    60: 'Usgao', 61: 'VAPI', 62: 'Vadodara', 63: 'Vapi'
}

# Reverse mapping for encoding
state_encoder = {v: k for k, v in STATE_MAPPING.items()}
type_encoder = {v: k for k, v in TYPE_MAPPING.items()}
loc_encoder = {v: k for k, v in LOC_MAPPING.items()}

encoders = {
    'state': state_encoder,
    'type': type_encoder,
    'location': loc_encoder
}

# Also ensure target dir exists
os.makedirs('models', exist_ok=True)

with open('models/encoder.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("Encoders saved successfully!")
