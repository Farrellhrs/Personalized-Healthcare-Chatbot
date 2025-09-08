import logging
from typing import List, Dict, Any
from database_handler import DatabaseHandler
from llm_handler import LLMHandler

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Generate personalized question recommendations for users"""
    
    def __init__(self, base_path):
        self.base_path = base_path
        
        # Default recommendations by category - FOCUSED ON HISTORICAL DATA QUERIES
        self.default_recommendations = {
            'lab_results': [
                "Apa hasil lab terakhir saya?",
                "Tampilkan riwayat hasil lab 3 bulan terakhir",
                "Bagaimana tren nilai lab dari kunjungan sebelumnya?"
            ],
            'diagnosis': [
                "Tampilkan riwayat diagnosis yang pernah saya terima",
                "Apa catatan diagnosis dari kunjungan terakhir?",
                "Bagaimana perbandingan diagnosis dari waktu ke waktu?"
            ],
            'appointments': [
                "Tampilkan jadwal kunjungan ANC yang sudah dilakukan",
                "Kapan kunjungan terakhir saya ke dokter?",
                "Berapa kali sudah kontrol kehamilan?"
            ],
            'medications': [
                "Obat apa yang pernah diresepkan untuk saya?",
                "Tampilkan riwayat suplemen yang sudah dikonsumsi",
                "Apa catatan pengobatan dari kunjungan sebelumnya?"
            ],
            'pregnancy': [
                "Tampilkan data perkembangan kehamilan dari catatan ANC",
                "Bagaimana riwayat kondisi fisik selama kehamilan ini?",
                "Apa saja hasil pemeriksaan kehamilan yang sudah tercatat?"
            ],
            'general': [
                "Golongan darah saya apa?",
                "Siapa dokter yang biasa menangani saya?",
                "Tampilkan ringkasan data kesehatan saya"
            ]
        }
    
    def generate_recommendations(self, customer_id: str, 
                               database_handler: DatabaseHandler,
                               llm_handler: LLMHandler) -> List[str]:
        """
        Generate personalized recommendations based on user's history
        
        Args:
            customer_id (str): Customer ID
            database_handler (DatabaseHandler): Database handler instance
            llm_handler (LLMHandler): LLM handler instance
            
        Returns:
            List[str]: List of 4 recommended questions
        """
        try:
            # Get user's history summary
            user_history = database_handler.get_user_history_summary(customer_id)
            
            # Get customer info for personalization
            customer_info = database_handler.tables.get('customer', None)
            customer_name = "Customer"
            
            if customer_info is not None:
                customer_row = customer_info[customer_info['customer_id'] == customer_id]
                if not customer_row.empty:
                    customer_name = customer_row.iloc[0]['name']
            
            # Try to generate recommendations using LLM
            try:
                llm_recommendations = llm_handler.generate_recommendations(
                    user_history, customer_name
                )
                
                if llm_recommendations and len(llm_recommendations) >= 4:
                    logger.info(f"Generated LLM recommendations for customer {customer_id}")
                    return llm_recommendations[:4]
                    
            except Exception as e:
                logger.warning(f"LLM recommendation generation failed: {str(e)}")
            
            # Fallback to rule-based recommendations
            logger.info(f"Using rule-based recommendations for customer {customer_id}")
            return self._generate_rule_based_recommendations(user_history, customer_id, database_handler)
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            # Ultimate fallback
            return self._get_default_recommendations()
    
    def _generate_rule_based_recommendations(self, user_history: Dict[str, Any],
                                           customer_id: str,
                                           database_handler: DatabaseHandler) -> List[str]:
        """Generate recommendations based on rules and available data"""
        try:
            recommendations = []
            
            # Check what data is available and prioritize recommendations
            
            # 1. Lab results recommendation - FOCUS ON HISTORICAL DATA
            if user_history.get('recent_lab_results'):
                latest_lab = user_history['recent_lab_results'][0]
                test_type = latest_lab.get('test_type', 'lab')
                recommendations.append(f"Tampilkan tren hasil {test_type} dari kunjungan sebelumnya")
            else:
                recommendations.append("Apakah ada catatan hasil lab dalam riwayat saya?")
            
            # 2. Diagnosis recommendation - FOCUS ON HISTORICAL RECORDS
            if user_history.get('recent_diagnoses'):
                recommendations.append("Tampilkan catatan diagnosis dari kunjungan-kunjungan sebelumnya")
            else:
                recommendations.append("Apakah ada riwayat diagnosis yang tercatat untuk saya?")
            
            # 3. Appointment/schedule recommendation
            # Check for ANC visits using proper relationship
            has_anc_visits = False
            if 'kehamilan' in database_handler.tables and 'anc_kunjungan' in database_handler.tables:
                # Get pregnancy IDs for this customer
                pregnancy_data = database_handler.tables['kehamilan'][
                    database_handler.tables['kehamilan']['customer_id'] == customer_id
                ]
                if not pregnancy_data.empty:
                    pregnancy_ids = pregnancy_data['id_kehamilan'].tolist()
                    anc_visits = database_handler.tables['anc_kunjungan'][
                        database_handler.tables['anc_kunjungan']['id_kehamilan'].isin(pregnancy_ids)
                    ]
                    has_anc_visits = not anc_visits.empty
            
            if has_anc_visits:
                recommendations.append("Tampilkan riwayat kunjungan ANC yang sudah dilakukan")
            else:
                # Check if user has any visits at all
                if user_history.get('recent_visits'):
                    recommendations.append("Apakah ada catatan kunjungan kesehatan sebelumnya?")
                else:
                    recommendations.append("Siapa dokter yang tersedia untuk konsultasi?")
            
            # 4. Medication/prescription recommendation - FOCUS ON HISTORICAL RECORDS
            if user_history.get('recent_visits'):
                recommendations.append("Tampilkan riwayat obat yang pernah diresepkan")
            else:
                # Check if user has any pregnancy-related data
                if 'kehamilan' in database_handler.tables:
                    pregnancy = database_handler.tables['kehamilan'][
                        database_handler.tables['kehamilan']['customer_id'] == customer_id
                    ]
                    if not pregnancy.empty:
                        recommendations.append("Apakah ada catatan suplemen kehamilan yang pernah dikonsumsi?")
                    else:
                        recommendations.append("Golongan darah saya apa?")
                else:
                    recommendations.append("Siapa dokter yang biasa menangani saya?")
            
            # Ensure we have 4 recommendations
            while len(recommendations) < 4:
                default_recs = self._get_default_recommendations()
                for rec in default_recs:
                    if rec not in recommendations:
                        recommendations.append(rec)
                        break
            
            return recommendations[:4]
            
        except Exception as e:
            logger.error(f"Error in rule-based recommendation generation: {str(e)}")
            return self._get_default_recommendations()
    
    def _get_default_recommendations(self) -> List[str]:
        """Get default recommendations when personalization fails - SAFE HISTORICAL DATA QUERIES"""
        return [
            "Tampilkan ringkasan data kesehatan saya",
            "Apakah ada catatan kunjungan terakhir?",
            "Siapa dokter yang biasa menangani saya?",
            "Golongan darah saya apa?"
        ]
    
    def get_contextual_recommendations(self, intent: str, customer_id: str,
                                     database_handler: DatabaseHandler,
                                     user_input: str = "",
                                     response_content: str = "") -> List[str]:
        """
        Get contextual recommendations based on current intent and conversation context
        
        Args:
            intent (str): Current classified intent
            customer_id (str): Customer ID
            database_handler (DatabaseHandler): Database handler
            user_input (str): User's original question
            response_content (str): Bot's response content
            
        Returns:
            List[str]: Contextual recommendations related to current conversation
        """
        try:
            recommendations = []
            
            # Intent-specific contextual recommendations based on conversation flow
            if intent in ['hasil_lab_ringkasan', 'hasil_lab_detail']:
                recommendations.extend([
                    "Bagaimana tren hasil lab saya dari waktu ke waktu?",
                    "Apakah ada nilai lab yang perlu diperhatikan?",
                    "Kapan sebaiknya melakukan tes lab selanjutnya?",
                    "Siapa dokter yang menangani hasil lab saya?"
                ])
                
            elif intent in ['riwayat_diagnosis', 'detail_diagnosis']:
                recommendations.extend([
                    "Bagaimana perbandingan diagnosis saya dengan kunjungan sebelumnya?",
                    "Apa tindakan yang diberikan untuk diagnosis ini?",
                    "Obat apa yang diresepkan untuk kondisi ini?",
                    "Kapan kontrol berikutnya untuk diagnosis ini?"
                ])
                
            elif intent == 'jadwal_dokter':
                recommendations.extend([
                    "Siapa dokter spesialis yang tersedia untuk konsultasi?",
                    "Bagaimana riwayat kunjungan saya ke dokter ini?",
                    "Apa jadwal praktik dokter minggu ini?",
                    "Di mana lokasi praktik dokter yang biasa saya kunjungi?"
                ])
                
            elif intent in ['riwayat_preskripsi_obat', 'detail_preskripsi_obat']:
                recommendations.extend([
                    "Berapa lama saya sudah mengonsumsi obat ini?",
                    "Apa efek samping yang perlu diperhatikan?",
                    "Kapan jadwal minum obat selanjutnya?",
                    "Apakah ada interaksi dengan obat lain yang saya konsumsi?"
                ])
                
            elif intent == 'anc_tracker':
                recommendations.extend([
                    "Bagaimana perkembangan berat badan dari kunjungan ANC sebelumnya?",
                    "Apakah tekanan darah saya dalam rentang normal?",
                    "Bagaimana perkembangan detak jantung janin?",
                    "Kapan jadwal ANC berikutnya?"
                ])
                
            elif intent == 'reminder_kontrol_kehamilan':
                recommendations.extend([
                    "Apa saja yang perlu dipersiapkan untuk kontrol berikutnya?",
                    "Bagaimana riwayat kunjungan ANC saya sejauh ini?",
                    "Apakah ada keluhan yang perlu saya sampaikan nanti?",
                    "Dimana lokasi praktik bidan untuk kontrol?"
                ])
                
            elif intent == 'riwayat_persalinan':
                recommendations.extend([
                    "Bagaimana kondisi bayi saat lahir?",
                    "Apakah ada komplikasi saat persalinan?",
                    "Bagaimana perbandingan dengan kehamilan sebelumnya?",
                    "Apa yang perlu dipersiapkan jika hamil lagi?"
                ])
                
            elif intent == 'imunisasi_tracker':
                recommendations.extend([
                    "Imunisasi apa saja yang sudah saya terima?",
                    "Kapan jadwal imunisasi berikutnya?",
                    "Apakah ada efek samping yang perlu diperhatikan?",
                    "Dimana saya bisa mendapat imunisasi lanjutan?"
                ])
                
            elif intent == 'riwayat_suplemen_kehamilan':
                recommendations.extend([
                    "Berapa lama saya sudah mengonsumsi suplemen ini?",
                    "Apa manfaat suplemen yang saya konsumsi?",
                    "Apakah dosis suplemen sudah sesuai?",
                    "Kapan sebaiknya mengganti jenis suplemen?"
                ])
                
            elif intent == 'panduan_persiapan_persalinan':
                recommendations.extend([
                    "Apa saja tanda-tanda akan melahirkan?",
                    "Bagaimana cara mengatasi kontraksi?",
                    "Apa yang harus dibawa ke rumah sakit?",
                    "Kapan sebaiknya ke rumah sakit saat kontraksi?"
                ])
                
            elif intent == 'cek_data_customer':
                recommendations.extend([
                    "Apakah data kontak saya masih aktual?",
                    "Bagaimana cara memperbarui data pribadi?",
                    "Siapa yang bisa dihubungi dalam keadaan darurat?",
                    "Apakah alamat saya sudah sesuai?"
                ])
                
            elif intent == 'cek_golongan_darah':
                recommendations.extend([
                    "Apa risiko golongan darah saya selama kehamilan?",
                    "Apakah pasangan perlu cek golongan darah juga?",
                    "Bagaimana cara menjaga kesehatan dengan golongan darah saya?",
                    "Kapan terakhir cek golongan darah?"
                ])
                
            elif intent == 'riwayat_berobat':
                recommendations.extend([
                    "Bagaimana tren kondisi kesehatan saya?",
                    "Apa diagnosis yang paling sering muncul?",
                    "Dokter mana yang paling sering menangani saya?",
                    "Kapan terakhir saya berobat untuk keluhan serupa?"
                ])
                
            else:
                # General contextual recommendations for unknown intents
                recommendations.extend([
                    "Bagaimana kondisi kesehatan saya secara keseluruhan?",
                    "Ada apa saja catatan medis terbaru untuk saya?",
                    "Kapan jadwal kontrol kesehatan berikutnya?",
                    "Siapa dokter yang biasa menangani saya?"
                ])
            
            # Enhance recommendations based on user input and response content
            enhanced_recommendations = self._enhance_recommendations_with_context(
                recommendations, user_input, response_content, intent, customer_id, database_handler
            )
            
            # Return 4 most relevant recommendations
            return enhanced_recommendations[:4]
            
        except Exception as e:
            logger.error(f"Error getting contextual recommendations: {str(e)}")
            return self._get_fallback_contextual_recommendations(intent)
    
    def _enhance_recommendations_with_context(self, base_recommendations: List[str],
                                            user_input: str, response_content: str,
                                            intent: str, customer_id: str,
                                            database_handler: DatabaseHandler) -> List[str]:
        """
        Enhance recommendations based on conversation context and available data
        """
        try:
            enhanced_recs = []
            
            # Analyze user input and response for keywords to personalize recommendations
            user_lower = user_input.lower() if user_input else ""
            response_lower = response_content.lower() if response_content else ""
            
            # Check what data is actually available for this customer
            available_data = self._check_available_data(customer_id, database_handler)
            
            for rec in base_recommendations:
                # Skip recommendations for data that doesn't exist
                if not self._is_recommendation_relevant(rec, available_data):
                    continue
                
                # Personalize recommendations based on context
                if "hasil lab" in rec.lower() and available_data.get('has_lab_results'):
                    enhanced_recs.append(rec)
                elif "diagnosis" in rec.lower() and available_data.get('has_diagnosis'):
                    enhanced_recs.append(rec)
                elif "anc" in rec.lower() and available_data.get('has_anc_visits'):
                    enhanced_recs.append(rec)
                elif "obat" in rec.lower() and available_data.get('has_prescriptions'):
                    enhanced_recs.append(rec)
                elif "suplemen" in rec.lower() and available_data.get('has_supplements'):
                    enhanced_recs.append(rec)
                elif "imunisasi" in rec.lower() and available_data.get('has_immunizations'):
                    enhanced_recs.append(rec)
                elif "persalinan" in rec.lower() and available_data.get('has_deliveries'):
                    enhanced_recs.append(rec)
                elif "dokter" in rec.lower():  # Always relevant
                    enhanced_recs.append(rec)
                else:
                    # Add generic recommendations that don't require specific data
                    if any(keyword in rec.lower() for keyword in ['jadwal', 'kontrol', 'kondisi', 'data', 'golongan darah']):
                        enhanced_recs.append(rec)
            
            # If we don't have enough relevant recommendations, add some general ones
            if len(enhanced_recs) < 4:
                general_recs = self._get_general_followup_recommendations(intent, available_data)
                for rec in general_recs:
                    if rec not in enhanced_recs and len(enhanced_recs) < 4:
                        enhanced_recs.append(rec)
            
            return enhanced_recs
            
        except Exception as e:
            logger.error(f"Error enhancing recommendations: {str(e)}")
            return base_recommendations
    
    def _check_available_data(self, customer_id: str, database_handler: DatabaseHandler) -> Dict[str, bool]:
        """Check what data is available for the customer"""
        try:
            available_data = {}
            
            # Check pregnancy-related data
            if 'kehamilan' in database_handler.tables:
                pregnancy_data = database_handler.tables['kehamilan'][
                    database_handler.tables['kehamilan']['customer_id'] == customer_id
                ]
                available_data['has_pregnancy'] = not pregnancy_data.empty
                
                if not pregnancy_data.empty:
                    pregnancy_ids = pregnancy_data['id_kehamilan'].tolist()
                    
                    # Check ANC visits
                    if 'anc_kunjungan' in database_handler.tables:
                        anc_visits = database_handler.tables['anc_kunjungan'][
                            database_handler.tables['anc_kunjungan']['id_kehamilan'].isin(pregnancy_ids)
                        ]
                        available_data['has_anc_visits'] = not anc_visits.empty
                    
                    # Check immunizations
                    if 'imunisasi_ibu_hamil' in database_handler.tables:
                        immunizations = database_handler.tables['imunisasi_ibu_hamil'][
                            database_handler.tables['imunisasi_ibu_hamil']['id_kehamilan'].isin(pregnancy_ids)
                        ]
                        available_data['has_immunizations'] = not immunizations.empty
                    
                    # Check deliveries
                    if 'persalinan' in database_handler.tables:
                        deliveries = database_handler.tables['persalinan'][
                            database_handler.tables['persalinan']['id_kehamilan'].isin(pregnancy_ids)
                        ]
                        available_data['has_deliveries'] = not deliveries.empty
            
            # Check general medical data
            if 'hasil_lab' in database_handler.tables:
                lab_results = database_handler.tables['hasil_lab'][
                    database_handler.tables['hasil_lab']['customer_id'] == customer_id
                ]
                available_data['has_lab_results'] = not lab_results.empty
            
            if 'riwayat_berobat' in database_handler.tables:
                visits = database_handler.tables['riwayat_berobat'][
                    database_handler.tables['riwayat_berobat']['customer_id'] == customer_id
                ]
                available_data['has_visits'] = not visits.empty
                
                if not visits.empty:
                    visit_ids = visits['visit_id'].tolist()
                    
                    # Check diagnoses
                    if 'diagnosis' in database_handler.tables:
                        diagnoses = database_handler.tables['diagnosis'][
                            database_handler.tables['diagnosis']['visit_id'].isin(visit_ids)
                        ]
                        available_data['has_diagnosis'] = not diagnoses.empty
                    
                    # Check prescriptions
                    if 'preskripsi' in database_handler.tables:
                        prescriptions = database_handler.tables['preskripsi'][
                            database_handler.tables['preskripsi']['visit_id'].isin(visit_ids)
                        ]
                        available_data['has_prescriptions'] = not prescriptions.empty
            
            return available_data
            
        except Exception as e:
            logger.error(f"Error checking available data: {str(e)}")
            return {}
    
    def _is_recommendation_relevant(self, recommendation: str, available_data: Dict[str, bool]) -> bool:
        """Check if a recommendation is relevant based on available data"""
        rec_lower = recommendation.lower()
        
        if 'lab' in rec_lower and not available_data.get('has_lab_results', False):
            return False
        elif 'diagnosis' in rec_lower and not available_data.get('has_diagnosis', False):
            return False
        elif 'anc' in rec_lower and not available_data.get('has_anc_visits', False):
            return False
        elif 'obat' in rec_lower and not available_data.get('has_prescriptions', False):
            return False
        elif 'imunisasi' in rec_lower and not available_data.get('has_immunizations', False):
            return False
        elif 'persalinan' in rec_lower and not available_data.get('has_deliveries', False):
            return False
        
        return True
    
    def _get_general_followup_recommendations(self, intent: str, available_data: Dict[str, bool]) -> List[str]:
        """Get general follow-up recommendations based on available data"""
        recommendations = []
        
        if available_data.get('has_pregnancy'):
            recommendations.extend([
                "Bagaimana perkembangan kehamilan saya secara keseluruhan?",
                "Ada apa saja catatan penting dari pemeriksaan kehamilan?"
            ])
        
        if available_data.get('has_visits'):
            recommendations.extend([
                "Siapa dokter yang paling sering menangani saya?",
                "Bagaimana tren kondisi kesehatan saya?"
            ])
        
        # Always relevant recommendations
        recommendations.extend([
            "Apa data kontak dokter atau bidan saya?",
            "Golongan darah saya apa?",
            "Ada apa saja informasi penting dalam data kesehatan saya?"
        ])
        
        return recommendations
    
    def _get_fallback_contextual_recommendations(self, intent: str) -> List[str]:
        """Get fallback contextual recommendations when errors occur"""
        return [
            "Bagaimana kondisi kesehatan saya secara umum?",
            "Apa saja catatan medis terbaru?",
            "Siapa dokter yang menangani saya?",
            "Golongan darah saya apa?"
        ]
