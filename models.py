"""
Mock models for MatCare - Maternal and Fetal Health Prediction Platform
"""
import numpy as np

class MockPregnancyModel:
    """Mock pregnancy risk prediction model"""
    
    def predict(self, X):
        """
        Predict pregnancy risk based on input parameters.
        Args:
            X: numpy array with shape (1, 5) containing [age, bp, sugar, temp, hr]
        Returns:
            numpy array with risk level: 0=Low, 1=Medium, 2=High
        """
        age, bp, sugar, temp, hr = X[0]
        
        risk_score = 0
        
        # Age risk factors
        if age < 18 or age > 40:
            risk_score += 2
        elif age < 20 or age > 35:
            risk_score += 1
            
        # Blood pressure risk
        if bp > 90:
            risk_score += 2
        elif bp > 85:
            risk_score += 1
            
        # Blood sugar risk
        if sugar > 7.8:
            risk_score += 2
        elif sugar > 6.5:
            risk_score += 1
            
        # Temperature and heart rate additional risk
        if temp > 37.5 or hr > 100:
            risk_score += 1
            
        # Convert score to risk level
        if risk_score >= 4:
            return np.array([2])  # High risk
        elif risk_score >= 2:
            return np.array([1])  # Medium risk
        else:
            return np.array([0])  # Low risk

class MockFetalModel:
    """Mock fetal health prediction model"""
    
    def predict(self, X):
        """
        Predict fetal health based on CTG parameters.
        Args:
            X: numpy array with shape (1, 21) containing CTG features
        Returns:
            numpy array with health status: 0=Normal, 1=Suspect, 2=Pathological
        """
        features = X[0]
        
        baseline_fhr = features[0] if len(features) > 0 else 140
        accelerations = features[1] if len(features) > 1 else 0
        decelerations = sum(features[4:7]) if len(features) > 6 else 0
        
        risk_score = 0
        
        # Baseline FHR assessment
        if baseline_fhr < 110 or baseline_fhr > 160:
            risk_score += 2
        elif baseline_fhr < 120 or baseline_fhr > 150:
            risk_score += 1
            
        # Accelerations assessment
        if accelerations < 0.001:
            risk_score += 1
            
        # Decelerations assessment
        if decelerations > 0.005:
            risk_score += 3
        elif decelerations > 0.002:
            risk_score += 2
        elif decelerations > 0:
            risk_score += 1
            
        # Convert score to health status
        if risk_score >= 5:
            return np.array([2])  # Pathological
        elif risk_score >= 2:
            return np.array([1])  # Suspect
        else:
            return np.array([0])  # Normal