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
                                     database_handler: DatabaseHandler) -> List[str]:
        """
        Get contextual recommendations based on current intent
        
        Args:
            intent (str): Current classified intent
            customer_id (str): Customer ID
            database_handler (DatabaseHandler): Database handler
            
        Returns:
            List[str]: Contextual recommendations
        """
        try:
            recommendations = []
            
            # Intent-specific recommendations - FOCUS ON HISTORICAL DATA ONLY
            if intent in ['hasil_lab_ringkasan', 'hasil_lab_detail']:
                recommendations.extend([
                    "Tampilkan tren hasil lab dari 3 bulan terakhir",
                    "Apakah ada perubahan nilai lab dari kunjungan sebelumnya?",
                    "Kapan terakhir kali melakukan tes lab?"
                ])
                
            elif intent in ['riwayat_diagnosis', 'detail_diagnosis']:
                recommendations.extend([
                    "Tampilkan catatan diagnosis dari kunjungan-kunjungan sebelumnya",
                    "Apakah ada perubahan diagnosis dari waktu ke waktu?",
                    "Kapan terakhir menerima diagnosis baru?"
                ])
                
            elif intent == 'jadwal_dokter':
                recommendations.extend([
                    "Siapa dokter spesialis yang biasa menangani saya?",
                    "Tampilkan riwayat kunjungan ke dokter tertentu",
                    "Di mana lokasi praktik dokter yang biasa saya datangi?"
                ])
                
            elif intent in ['riwayat_preskripsi_obat', 'detail_preskripsi_obat']:
                recommendations.extend([
                    "Tampilkan riwayat obat yang pernah diresepkan",
                    "Apa catatan dosis obat dari resep sebelumnya?",
                    "Kapan terakhir mendapat resep obat baru?"
                ])
                
            elif intent == 'anc_tracker':
                recommendations.extend([
                    "Tampilkan tren berat badan dari kunjungan ANC sebelumnya",
                    "Bagaimana riwayat tekanan darah selama kehamilan ini?",
                    "Apakah ada catatan detak jantung janin dari pemeriksaan sebelumnya?"
                ])
                
            # Fill remaining slots with general recommendations
            general_recs = self._get_default_recommendations()
            for rec in general_recs:
                if len(recommendations) < 4 and rec not in recommendations:
                    recommendations.append(rec)
            
            return recommendations[:4]
            
        except Exception as e:
            logger.error(f"Error getting contextual recommendations: {str(e)}")
            return self._get_default_recommendations()
