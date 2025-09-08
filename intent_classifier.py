import pandas as pd
import numpy as np
import os
import json
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError:
    from transformers.models.auto.tokenization_auto import AutoTokenizer
    from transformers.models.auto.modeling_auto import AutoModelForSequenceClassification
import pickle

logger = logging.getLogger(__name__)

class IntentClassifier:
    """Handle intent classification using embeddings and BERT models"""
    
    def __init__(self, base_path, similarity_threshold=0.6):
        self.base_path = base_path
        self.similarity_threshold = similarity_threshold
        
        # Load intent data
        self.intent_file = os.path.join(base_path, "intent_merged.csv")
        self._load_intent_data()
        
        # Initialize sentence transformer for similarity checking
        self._initialize_sentence_transformer()
        
        # Load BERT models
        self.bert_models_path = os.path.join(base_path, "Model BERT")
        self._load_bert_models()
    
    def _load_intent_data(self):
        """Load intent data from CSV"""
        try:
            if not os.path.exists(self.intent_file):
                raise FileNotFoundError(f"Intent file not found: {self.intent_file}")
            
            self.intents_df = pd.read_csv(self.intent_file)
            logger.info(f"Loaded {len(self.intents_df)} intent examples")
            
            # Get unique intents
            self.unique_intents = self.intents_df['intent'].unique()
            logger.info(f"Found {len(self.unique_intents)} unique intents")
            
        except Exception as e:
            logger.error(f"Error loading intent data: {str(e)}")
            raise
    
    def _initialize_sentence_transformer(self):
        """Initialize sentence transformer for similarity checking"""
        try:
            # Use the same model as in test_hamil.ipynb
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Group training examples by intent (same as test_hamil.ipynb)
            logger.info("Computing embeddings for intent examples...")
            self.training_examples_by_intent = {}
            for _, row in self.intents_df.iterrows():
                intent = row['intent']
                text = row['text']
                
                if intent not in self.training_examples_by_intent:
                    self.training_examples_by_intent[intent] = []
                self.training_examples_by_intent[intent].append(text)
            
            # Compute embeddings for all training examples grouped by intent
            self.training_embeddings_by_intent = {}
            for intent, texts in self.training_examples_by_intent.items():
                logger.info(f"Computing embeddings for '{intent}' ({len(texts)} examples)...")
                embeddings = self.sentence_model.encode(texts)
                self.training_embeddings_by_intent[intent] = embeddings
            
            logger.info("Sentence transformer initialized successfully")
            logger.info(f"Total intents: {len(self.training_embeddings_by_intent)}")
            
        except Exception as e:
            logger.error(f"Error initializing sentence transformer: {str(e)}")
            raise
    
    def _load_bert_models(self):
        """Load BERT models for domain and intent classification"""
        try:
            # Load domain classifier (hamil_umum)
            domain_model_path = os.path.join(self.bert_models_path, "model_hamil_umum")
            if os.path.exists(domain_model_path):
                self.domain_tokenizer = AutoTokenizer.from_pretrained(domain_model_path)
                self.domain_model = AutoModelForSequenceClassification.from_pretrained(domain_model_path)
                self.domain_model.eval()
                logger.info("Domain classifier loaded successfully")
            else:
                logger.warning(f"Domain model not found at {domain_model_path}")
                self.domain_tokenizer = None
                self.domain_model = None
            
            # Load pregnancy intent classifier
            pregnancy_model_path = os.path.join(self.bert_models_path, "model_hamil")
            if os.path.exists(pregnancy_model_path):
                self.pregnancy_tokenizer = AutoTokenizer.from_pretrained(pregnancy_model_path)
                self.pregnancy_model = AutoModelForSequenceClassification.from_pretrained(pregnancy_model_path)
                self.pregnancy_model.eval()
                
                # Load label encoder for pregnancy model
                label_encoder_path = os.path.join(pregnancy_model_path, "label_encoder.pkl")
                if os.path.exists(label_encoder_path):
                    with open(label_encoder_path, "rb") as f:
                        self.pregnancy_label_encoder = pickle.load(f)
                    logger.info(f"Pregnancy label encoder loaded with classes: {self.pregnancy_label_encoder.classes_}")
                else:
                    logger.warning(f"Label encoder not found at: {label_encoder_path}")
                    self.pregnancy_label_encoder = None
                
                # Load metadata for additional info
                metadata_path = os.path.join(pregnancy_model_path, "training_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r", encoding='utf-8') as f:
                        self.pregnancy_metadata = json.load(f)
                    logger.info(f"Pregnancy model metadata loaded: {self.pregnancy_metadata['num_classes']} classes")
                else:
                    self.pregnancy_metadata = {}
                
                logger.info("Pregnancy intent classifier loaded successfully")
            else:
                logger.warning(f"Pregnancy model not found at {pregnancy_model_path}")
                self.pregnancy_tokenizer = None
                self.pregnancy_model = None
                self.pregnancy_label_encoder = None
                self.pregnancy_metadata = {}
            
            # Load general intent classifier
            general_model_path = os.path.join(self.bert_models_path, "model_umum")
            if os.path.exists(general_model_path):
                self.general_tokenizer = AutoTokenizer.from_pretrained(general_model_path)
                self.general_model = AutoModelForSequenceClassification.from_pretrained(general_model_path)
                self.general_model.eval()
                logger.info("General intent classifier loaded successfully")
            else:
                logger.warning(f"General model not found at {general_model_path}")
                self.general_tokenizer = None
                self.general_model = None
                
        except Exception as e:
            logger.error(f"Error loading BERT models: {str(e)}")
            # Set models to None if loading fails
            self.domain_tokenizer = None
            self.domain_model = None
            self.pregnancy_tokenizer = None
            self.pregnancy_model = None
            self.pregnancy_label_encoder = None
            self.pregnancy_metadata = {}
            self.general_tokenizer = None
            self.general_model = None
    
    def check_similarity(self, user_input):
        """
        Check similarity between user input and intent examples
        
        Args:
            user_input (str): User's input text
            
        Returns:
            dict: Similarity check result with is_valid flag and best matches
        """
        try:
            # Encode user input
            user_embedding = self.sentence_model.encode([user_input])
            
            # Find best matches across all intents
            all_similarities = []
            
            for intent, training_embeddings in self.training_embeddings_by_intent.items():
                # Compute similarities with all training examples of this intent
                similarities = cosine_similarity(user_embedding, training_embeddings)[0]
                max_similarity_for_intent = float(np.max(similarities))
                
                all_similarities.append({
                    'intent': intent,
                    'max_similarity': max_similarity_for_intent,
                    'mean_similarity': float(np.mean(similarities))
                })
            
            # Sort by max similarity
            all_similarities.sort(key=lambda x: x['max_similarity'], reverse=True)
            
            # Get best matches (top 5)
            best_matches = all_similarities[:5]
            
            # Check if highest similarity meets threshold
            max_similarity = best_matches[0]['max_similarity'] if best_matches else 0.0
            is_valid = max_similarity >= self.similarity_threshold
            
            logger.info(f"Similarity check - Max: {max_similarity:.3f}, Threshold: {self.similarity_threshold}, Valid: {is_valid}")
            
            return {
                'is_valid': is_valid,
                'max_similarity': max_similarity,
                'best_matches': best_matches
            }
            
        except Exception as e:
            logger.error(f"Error checking similarity: {str(e)}")
            return {'is_valid': False, 'max_similarity': 0.0, 'best_matches': []}
    
    def classify_domain(self, user_input):
        """
        Classify whether input is KEHAMILAN or UMUM domain
        
        Args:
            user_input (str): User's input text
            
        Returns:
            str: 'KEHAMILAN' or 'UMUM'
        """
        try:
            if self.domain_model is None or self.domain_tokenizer is None:
                # Fallback: simple keyword-based classification
                pregnancy_keywords = ['hamil', 'kehamilan', 'kandungan', 'anc', 'persalinan', 
                                    'melahirkan', 'trimester', 'kontraksi', 'janin', 'bayi']
                user_lower = user_input.lower()
                
                for keyword in pregnancy_keywords:
                    if keyword in user_lower:
                        return 'KEHAMILAN'
                
                return 'UMUM'
            
            # Use BERT model for classification
            inputs = self.domain_tokenizer(
                user_input,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.domain_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
            
            # Assuming class 0 is UMUM, class 1 is KEHAMILAN
            domain = 'UMUM' if predicted_class == 1 else 'KEHAMILAN'
            confidence = float(predictions.max())
            
            logger.info(f"Domain classification: {domain} (confidence: {confidence:.3f})")
            
            return domain
            
        except Exception as e:
            logger.error(f"Error in domain classification: {str(e)}")
            return 'UMUM'  # Default fallback
    
    def classify_pregnancy_intent(self, user_input):
        """
        Classify specific pregnancy-related intent
        
        Args:
            user_input (str): User's input text
            
        Returns:
            dict: Intent classification result
        """
        try:
            if self.pregnancy_model is None or self.pregnancy_tokenizer is None or self.pregnancy_label_encoder is None:
                # Fallback: use similarity-based approach
                return self._fallback_intent_classification(user_input, 'KEHAMILAN')
            
            # Use BERT model for pregnancy intent classification
            inputs = self.pregnancy_tokenizer(
                user_input,
                return_tensors="pt",
                max_length=128,  # Updated to match training config
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.pregnancy_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Get top 2 predictions
                top_predictions = torch.topk(predictions, 2, dim=-1)
                predicted_classes = top_predictions.indices[0].tolist()
                confidences = top_predictions.values[0].tolist()
            
            # Use label encoder to get actual class names
            # Based on training_metadata.json, the model has 6 classes:
            # anc_tracker=0, imunisasi_tracker=1, panduan_persiapan_persalinan=2,
            # reminder_kontrol_kehamilan=3, riwayat_persalinan=4, riwayat_suplemen_kehamilan=5
            
            # Create results for top 2 predictions
            results = []
            for i, (class_id, confidence) in enumerate(zip(predicted_classes, confidences)):
                if class_id < len(self.pregnancy_label_encoder.classes_):
                    intent = self.pregnancy_label_encoder.inverse_transform([class_id])[0]
                else:
                    intent = 'panduan_persiapan_persalinan'  # Default fallback
                
                results.append({
                    'intent': intent,
                    'confidence': float(confidence),
                    'rank': i + 1,
                    'class_id': class_id
                })
            
            logger.info(f"Pregnancy intent classification - Top prediction: {results[0]['intent']} (confidence: {results[0]['confidence']:.3f})")
            
            # Return both primary result (for backward compatibility) and top 2 predictions
            return {
                'intent': results[0]['intent'],
                'confidence': results[0]['confidence'],
                'predictions': results
            }
            
        except Exception as e:
            logger.error(f"Error in pregnancy intent classification: {str(e)}")
            return self._fallback_intent_classification(user_input, 'KEHAMILAN')
    
    def classify_general_intent(self, user_input):
        """
        Classify general (non-pregnancy) intent
        
        Args:
            user_input (str): User's input text
            
        Returns:
            dict: Intent classification result
        """
        try:
            if self.general_model is None or self.general_tokenizer is None:
                return self._fallback_intent_classification(user_input, 'UMUM')
            
            # Use BERT model for general intent classification
            inputs = self.general_tokenizer(
                user_input,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            with torch.no_grad():
                outputs = self.general_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
                # Get top 2 predictions
                top_predictions = torch.topk(predictions, 2, dim=-1)
                predicted_classes = top_predictions.indices[0].tolist()
                confidences = top_predictions.values[0].tolist()
            
            # Get general intents - CORRECTED to match training label mapping
            # IMPORTANT: Order must match exactly with training data label encoding:
            # cek_data_customer=0, cek_golongan_darah=1, detail_dokter=2, detail_preskripsi_obat=3,
            # hasil_lab_detail=4, hasil_lab_ringkasan=5, jadwal_dokter=6, riwayat_berobat=7,
            # riwayat_diagnosis=8, riwayat_kondisi_fisik=9, riwayat_preskripsi_obat=10
            general_intents = [
                'cek_data_customer',              # 0
                'cek_golongan_darah',            # 1
                'detail_dokter',                 # 2
                'detail_preskripsi_obat',        # 3
                'hasil_lab_detail',              # 4
                'hasil_lab_ringkasan',           # 5
                'jadwal_dokter',                 # 6
                'riwayat_berobat',              # 7
                'riwayat_diagnosis',            # 8
                'riwayat_kondisi_fisik',        # 9
                'riwayat_preskripsi_obat'       # 10
            ]
            
            # Create results for top 2 predictions
            results = []
            for i, (class_id, confidence) in enumerate(zip(predicted_classes, confidences)):
                if class_id < len(general_intents):
                    intent = general_intents[class_id]
                else:
                    intent = 'cek_data_customer'  # Default (first intent)
                
                results.append({
                    'intent': intent,
                    'confidence': float(confidence),
                    'rank': i + 1
                })
            
            # Return both primary result (for backward compatibility) and top 2 predictions
            return {
                'intent': results[0]['intent'],
                'confidence': results[0]['confidence'],
                'predictions': results
            }
            
        except Exception as e:
            logger.error(f"Error in general intent classification: {str(e)}")
            return self._fallback_intent_classification(user_input, 'UMUM')
    
    def _fallback_intent_classification(self, user_input, domain):
        """
        Fallback intent classification using similarity matching
        
        Args:
            user_input (str): User's input text
            domain (str): Domain ('KEHAMILAN' or 'UMUM')
            
        Returns:
            dict: Intent classification result
        """
        try:
            # Use the similarity results from check_similarity
            similarity_result = self.check_similarity(user_input)
            
            if similarity_result['best_matches']:
                # Get the best matching intent
                best_match = similarity_result['best_matches'][0]
                intent = best_match['intent']
                confidence = best_match['similarity']
                
                logger.info(f"Fallback classification: {intent} (confidence: {confidence:.3f})")
                
                return {
                    'intent': intent,
                    'confidence': confidence,
                    'predicted_class': 0
                }
            else:
                # Default intents based on domain - UPDATED after merger
                default_intent = 'panduan_persiapan_persalinan' if domain == 'KEHAMILAN' else 'cek_data_customer'
                
                return {
                    'intent': default_intent,
                    'confidence': 0.5,
                    'predicted_class': 0
                }
                
        except Exception as e:
            logger.error(f"Error in fallback classification: {str(e)}")
            default_intent = 'panduan_persiapan_persalinan' if domain == 'KEHAMILAN' else 'cek_data_customer'
            
            return {
                'intent': default_intent,
                'confidence': 0.1,
                'predicted_class': 0
            }
    
    def predict_intent_with_similarity(self, user_input, confidence_threshold=0.7, similarity_threshold=0.74):
        """
        Enhanced prediction using both classifier and similarity verification
        Similar to the method in test_hamil.ipynb
        
        Args:
            user_input (str): User's input text
            confidence_threshold (float): Minimum classifier confidence
            similarity_threshold (float): Minimum similarity score
            
        Returns:
            dict: Prediction results with decision logic
        """
        try:
            # Step 1: Classify domain
            domain = self.classify_domain(user_input)
            
            # Step 2: Get classifier prediction based on domain
            if domain == 'KEHAMILAN':
                classification_result = self.classify_pregnancy_intent(user_input)
            else:
                classification_result = self.classify_general_intent(user_input)
            
            predicted_intent = classification_result['intent']
            classifier_confidence = classification_result['confidence']
            top_predictions = classification_result.get('predictions', [])
            
            # Step 3: Compute similarity with training examples
            user_embedding = self.sentence_model.encode([user_input])
            
            # Get embeddings for the predicted intent
            if predicted_intent in self.training_embeddings_by_intent:
                training_embeddings = self.training_embeddings_by_intent[predicted_intent]
                
                # Compute cosine similarity with all training examples of predicted intent
                similarities = cosine_similarity(user_embedding, training_embeddings)[0]
                max_similarity = float(np.max(similarities))
                mean_similarity = float(np.mean(similarities))
            else:
                max_similarity = 0.0
                mean_similarity = 0.0
            
            # Step 4: Apply thresholding logic
            meets_confidence = classifier_confidence >= confidence_threshold
            meets_similarity = max_similarity >= similarity_threshold
            
            if meets_confidence and meets_similarity:
                final_decision = predicted_intent
                decision_reason = f"High confidence ({classifier_confidence:.3f}) and similarity ({max_similarity:.3f})"
            else:
                final_decision = "out_of_scope"
                if not meets_confidence and not meets_similarity:
                    decision_reason = f"Low confidence ({classifier_confidence:.3f}) and similarity ({max_similarity:.3f})"
                elif not meets_confidence:
                    decision_reason = f"Low confidence ({classifier_confidence:.3f}), similarity OK ({max_similarity:.3f})"
                else:  # not meets_similarity
                    decision_reason = f"Confidence OK ({classifier_confidence:.3f}), low similarity ({max_similarity:.3f})"
            
            logger.info(f"Enhanced prediction - Input: '{user_input}', Final: {final_decision}, Reason: {decision_reason}")
            
            return {
                'text': user_input,
                'domain': domain,
                'classifier_prediction': predicted_intent,
                'classifier_confidence': classifier_confidence,
                'top_predictions': top_predictions,  # Added top 2 predictions
                'max_similarity': max_similarity,
                'mean_similarity': mean_similarity,
                'final_decision': final_decision,
                'decision_reason': decision_reason,
                'confidence_threshold': confidence_threshold,
                'similarity_threshold': similarity_threshold,
                'meets_confidence': meets_confidence,
                'meets_similarity': meets_similarity
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced prediction: {str(e)}")
            return {
                'text': user_input,
                'domain': 'UMUM',
                'classifier_prediction': 'unknown',
                'classifier_confidence': 0.0,
                'max_similarity': 0.0,
                'mean_similarity': 0.0,
                'final_decision': 'out_of_scope',
                'decision_reason': f'Error during prediction: {str(e)}',
                'confidence_threshold': confidence_threshold,
                'similarity_threshold': similarity_threshold,
                'meets_confidence': False,
                'meets_similarity': False
            }
