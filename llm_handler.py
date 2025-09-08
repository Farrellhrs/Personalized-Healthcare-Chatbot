import logging
import json
from typing import Dict, Any, List
import google.generativeai as genai
import streamlit as st
from datetime import datetime, timedelta

from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class LLMHandler:
    """Handle LLM interactions with Google Gemini"""
    
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize Gemini model"""
        try:
            # Get API key from environment or streamlit secrets
            if not self.api_key:
                try:
                    self.api_key = st.secrets["GEMINI_API_KEY"]
                except:
                    # For development, you can set it here temporarily
                    # self.api_key = "your_gemini_api_key_here"
                    logger.warning("Gemini API key not found. Please set GEMINI_API_KEY in secrets.toml")
                    return
            
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
                logger.info("Gemini model initialized successfully")
            else:
                logger.error("No Gemini API key provided")
                
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            self.model = None
    
    def generate_response(self, user_input: str, intent: str, confidence: float, 
                         db_context: Dict[str, Any], user_data: Dict[str, Any], 
                         top_predictions: Optional[List[Dict[str, Any]]] = None,
                         contexts: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
        """
        Generate response using Gemini LLM with context
        
        Args:
            user_input (str): User's original question
            intent (str): Classified intent (primary)
            confidence (float): Intent classification confidence (primary)
            db_context (dict): Database context for the intent (backward compatibility)
            user_data (dict): User's basic information
            top_predictions (list): Top 2 predictions with confidence scores
            contexts (dict): All contexts for top predictions {'primary': {...}, 'prediction_1': {...}, 'prediction_2': {...}}
            
        Returns:
            str: Generated response
        """
        try:
            if not self.model:
                return "Maaf, sistem sedang mengalami gangguan. Silakan coba lagi nanti."
            
            # Special handling for ANC reminder - provide direct calculated response
            if intent == 'reminder_kontrol_kehamilan':
                return self._calculate_anc_reminder(db_context, user_data)
            
            # Build context prompt for other intents
            prompt = self._build_prompt(user_input, intent, confidence, db_context, user_data, top_predictions, contexts)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                return response.text.strip()
            else:
                return "Maaf, saya tidak dapat memberikan jawaban yang tepat saat ini."
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "Maaf, terjadi kesalahan dalam memproses pertanyaan Anda. Silakan coba lagi."
    
    def _build_prompt(self, user_input: str, intent: str, confidence: float,
                     db_context: Dict[str, Any], user_data: Dict[str, Any], 
                     top_predictions: Optional[List[Dict[str, Any]]] = None,
                     contexts: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
        """Build comprehensive prompt for Gemini"""
        
        # Special handling for ANC reminder intent
        if intent == 'reminder_kontrol_kehamilan':
            anc_reminder = self._calculate_anc_reminder(db_context, user_data)
            return f"""
Anda adalah asisten chatbot untuk sistem kesehatan ibu hamil. Pengguna menanyakan jadwal kontrol ANC berikutnya.

INFORMASI PENGGUNA:
- Nama: {user_data.get('name', 'N/A')}
- NIK: {user_data.get('NIK', 'N/A')}
- Customer ID: {user_data.get('customer_id', 'N/A')}

PERTANYAAN PENGGUNA: "{user_input}"

JADWAL ANC YANG DIHITUNG:
{anc_reminder}

INSTRUKSI:
- Sampaikan informasi jadwal ANC dengan jelas dan lengkap
- Gunakan bahasa Indonesia yang ramah dan mudah dipahami
- Tekankan pentingnya kontrol rutin untuk kesehatan ibu dan janin
- Jika ada keterlambatan, sampaikan dengan lembut namun tegas

Jawaban Anda:
"""
        
        # Build intent predictions context for other intents
        intent_context = f"INTENT YANG TERDETEKSI: {intent} (confidence: {confidence:.3f})"
        
        if top_predictions and len(top_predictions) > 1:
            # Add confidence gap analysis
            confidence_gap = top_predictions[0]['confidence'] - top_predictions[1]['confidence']
            
            intent_context += f"\n\nTOP 2 PREDIKSI:"
            for i, pred in enumerate(top_predictions, 1):
                intent_context += f"\n{i}. {pred['intent']} (confidence: {pred['confidence']:.3f})"
            
            intent_context += f"\n\nCONFIDENCE GAP: {confidence_gap:.3f}"
            
            if confidence_gap > 0.5:  # High confidence in primary intent
                intent_context += "\n→ TINGKAT KEYAKINAN TINGGI pada intent utama"
                context_instruction = "Fokus penuh pada PRIMARY INTENT untuk menjawab"
            elif confidence_gap > 0.2:  # Medium confidence gap
                intent_context += "\n→ TINGKAT KEYAKINAN MENENGAH pada intent utama"
                context_instruction = "Fokus pada PRIMARY INTENT, tapi pertimbangkan kemungkinan ALTERNATIVE INTENT jika relevan"
            else:  # Low confidence gap - ambiguous
                intent_context += "\n→ PERTANYAAN AMBIGU - confidence gap rendah"
                context_instruction = "Pertimbangkan kedua intent yang mungkin dan pilih jawaban yang paling relevan dengan pertanyaan"
        else:
            context_instruction = "Jawab berdasarkan intent yang terdeteksi"
        
        # Add intent-specific instructions
        intent_specific_instruction = self._get_intent_specific_instruction(intent)
        
        prompt = f"""
Anda adalah asisten chatbot untuk sistem kesehatan ibu hamil. Tugas Anda adalah menjawab pertanyaan pengguna berdasarkan data medis mereka dan knowledge base yang tersedia.

ATURAN KETAT:
1. HANYA jawab berdasarkan data yang disediakan dalam konteks
2. JANGAN berikan saran medis atau diagnosis
3. JANGAN buat asumsi tentang kondisi kesehatan
4. Jika data tidak tersedia, katakan dengan jelas
5. Gunakan bahasa Indonesia yang sopan dan ramah
6. Fokus pada informasi administratif dan faktual saja

INFORMASI PENGGUNA:
- Nama: {user_data.get('name', 'N/A')}
- NIK: {user_data.get('NIK', 'N/A')}
- Customer ID: {user_data.get('customer_id', 'N/A')}

PERTANYAAN PENGGUNA: "{user_input}"

{intent_context}

DESKRIPSI INTENT: {db_context.get('intent_description', 'N/A')}

{intent_specific_instruction}

DATA RELEVAN DARI DATABASE:
{self._format_multiple_contexts(db_context, contexts, top_predictions)}

KNOWLEDGE BASE:
{self._format_knowledge_base(db_context.get('knowledge_base', {}))}

INSTRUKSI JAWABAN:
- {context_instruction}
- Jawab dalam bahasa Indonesia
- Gunakan data yang tersedia di atas
- Jika data kosong atau tidak relevan, jelaskan dengan sopan
- Berikan informasi yang akurat dan mudah dipahami
- JANGAN memberikan saran medis, hanya informasi faktual
- Jika diminta saran medis, arahkan untuk konsultasi dengan dokter

Jawaban Anda:
"""
        
        return prompt
    
    def _format_multiple_contexts(self, primary_context: Dict[str, Any], 
                                contexts: Optional[Dict[str, Dict[str, Any]]] = None,
                                top_predictions: Optional[List[Dict[str, Any]]] = None) -> str:
        """Format multiple contexts for top predictions"""
        if not contexts or len(contexts) == 1:
            # Fallback to single context formatting
            return self._format_database_context(primary_context.get('data', {}))
        
        formatted_sections = []
        
        # Format primary intent context
        formatted_sections.append("=== DATA UNTUK INTENT UTAMA ===")
        formatted_sections.append(f"Intent: {primary_context.get('intent', 'N/A')}")
        formatted_sections.append(self._format_database_context(primary_context.get('data', {})))
        
        # Format alternative intent contexts
        if top_predictions and len(top_predictions) > 1:
            for i, prediction in enumerate(top_predictions[1:], 2):  # Start from 2nd prediction
                context_key = f"prediction_{i}"
                if context_key in contexts:
                    alt_context = contexts[context_key]
                    formatted_sections.append(f"\n=== DATA UNTUK INTENT ALTERNATIF #{i-1} ===")
                    formatted_sections.append(f"Intent: {prediction['intent']} (confidence: {prediction['confidence']:.3f})")
                    formatted_sections.append(self._format_database_context(alt_context.get('data', {})))
        
        return '\n'.join(formatted_sections) if formatted_sections else "Tidak ada data yang tersedia."
    
    def _format_database_context(self, data_context: Dict[str, Any]) -> str:
        """Format database context for the prompt"""
        if not data_context:
            return "Tidak ada data yang tersedia."
        
        formatted_sections = []
        
        for key, value in data_context.items():
            if isinstance(value, list) and value:
                formatted_sections.append(f"\n{key.upper()}:")
                for i, item in enumerate(value[:3]):  # Limit to 3 most recent items
                    if isinstance(item, dict):
                        item_info = []
                        for k, v in item.items():
                            if v and str(v).strip():
                                item_info.append(f"{k}: {v}")
                        formatted_sections.append(f"  {i+1}. {', '.join(item_info)}")
            elif isinstance(value, dict) and value:
                formatted_sections.append(f"\n{key.upper()}:")
                for k, v in value.items():
                    if v and str(v).strip():
                        formatted_sections.append(f"  {k}: {v}")
        
        return '\n'.join(formatted_sections) if formatted_sections else "Tidak ada data yang tersedia."
    
    def _format_knowledge_base(self, knowledge_base: Dict[str, Any]) -> str:
        """Format knowledge base context for the prompt"""
        if not knowledge_base:
            return "Tidak ada knowledge base yang relevan."
        
        formatted_sections = []
        
        for key, value in knowledge_base.items():
            if value and str(value).strip():
                formatted_sections.append(f"\n{key.upper()}:")
                # Truncate very long knowledge base content
                content = str(value)[:2000] + "..." if len(str(value)) > 2000 else str(value)
                formatted_sections.append(content)
        
        return '\n'.join(formatted_sections) if formatted_sections else "Tidak ada knowledge base yang relevan."
    
    def _get_intent_specific_instruction(self, intent: str) -> str:
        """Get specific instruction for each intent to guide the AI response"""
        intent_instructions = {
            'riwayat_persalinan': """
INSTRUKSI KHUSUS UNTUK RIWAYAT PERSALINAN:
- FOKUS UTAMA pada data DELIVERIES (persalinan) yang berisi informasi persalinan aktual
- Tampilkan: tanggal lahir, tempat lahir, cara persalinan, jenis kelamin bayi, berat bayi, panjang bayi, komplikasi
- Data PREGNANCIES hanya sebagai konteks kehamilan yang terkait
- Jika tidak ada data persalinan, jelaskan bahwa belum ada riwayat persalinan yang tercatat
- Jangan fokus hanya pada data kehamilan, tetapi prioritaskan informasi persalinan""",
            
            'anc_tracker': """
INSTRUKSI KHUSUS UNTUK ANC TRACKER:
- Tampilkan riwayat kunjungan ANC dengan detail: tanggal, usia kehamilan, berat badan, tekanan darah
- Sertakan informasi kehamilan sebagai konteks
- Urutkan dari kunjungan terbaru ke terlama""",
            
            'imunisasi_tracker': """
INSTRUKSI KHUSUS UNTUK IMUNISASI TRACKER:
- Fokus pada data imunisasi: jenis vaksin, tanggal pemberian, dosis
- Tampilkan jadwal imunisasi yang sudah dilakukan dan yang mungkin belum""",
            
            'riwayat_suplemen_kehamilan': """
INSTRUKSI KHUSUS UNTUK SUPLEMEN KEHAMILAN:
- Fokus pada data suplemen yang pernah dikonsumsi
- Tampilkan: nama suplemen, dosis, tanggal mulai konsumsi
- Hubungkan dengan kunjungan ANC yang terkait""",
            
            'reminder_kontrol_kehamilan': """
INSTRUKSI KHUSUS UNTUK REMINDER KONTROL:
- Hitung dan tampilkan jadwal kontrol berikutnya berdasarkan usia kehamilan
- Berikan panduan interval kontrol sesuai trimester""",
            
            'panduan_persiapan_persalinan': """
INSTRUKSI KHUSUS UNTUK PANDUAN PERSIAPAN PERSALINAN:
- Gunakan knowledge base untuk memberikan informasi umum tentang persiapan persalinan
- Tidak perlu data personal kecuali sebagai konteks umur kehamilan"""
        }
        
        return intent_instructions.get(intent, "Jawab sesuai dengan intent yang terdeteksi dan data yang tersedia.")
    
    def _calculate_anc_reminder(self, db_context: Dict[str, Any], user_data: Dict[str, Any]) -> str:
        """
        Calculate next ANC visit date based on WHO/Ministry of Health guidelines
        
        Args:
            db_context: Database context containing ANC visit history, pregnancy data, and delivery history
            user_data: User information
            
        Returns:
            str: Formatted ANC reminder message with next visit date
        """
        try:
            anc_visits = db_context.get('data', {}).get('anc_visits', [])
            pregnancy_data = db_context.get('data', {}).get('pregnancy_data', [])
            deliveries = db_context.get('data', {}).get('deliveries', [])
            
            # Check if there's any pregnancy data
            if not pregnancy_data:
                return """Belum ada data kehamilan yang tercatat untuk Anda. 
                
Jika Anda sedang hamil, silakan lakukan pendaftaran kehamilan dan kunjungan ANC pertama untuk memulai pemantauan kesehatan ibu dan janin."""

            # Get the most recent pregnancy
            current_pregnancy = pregnancy_data[0]  # Already sorted by latest
            pregnancy_id = current_pregnancy.get('id_kehamilan')
            pregnancy_status = current_pregnancy.get('status_kehamilan', '').lower()
            
            # Check if this pregnancy has already resulted in delivery
            pregnancy_delivered = False
            delivery_date = None
            if deliveries:
                for delivery in deliveries:
                    if delivery.get('id_kehamilan') == pregnancy_id:
                        pregnancy_delivered = True
                        delivery_date = delivery.get('tanggal_lahir')
                        break
            
            # If pregnancy has been delivered, no more ANC visits needed
            if pregnancy_delivered:
                return f"""Berdasarkan data Anda, kehamilan dengan ID {pregnancy_id} telah selesai dengan persalinan pada tanggal {delivery_date}.

Untuk kehamilan yang sudah selesai, tidak diperlukan lagi kontrol ANC. Jika Anda memiliki kehamilan baru, silakan lakukan pendaftaran kehamilan baru untuk memulai pemantauan ANC yang sesuai."""

            # Check pregnancy status
            if pregnancy_status not in ['berjalan', 'aktif', 'ongoing']:
                return f"""Status kehamilan Anda saat ini tercatat sebagai '{pregnancy_status}'. 

Silakan konsultasikan dengan bidan atau dokter mengenai status kehamilan Anda dan jadwal kontrol yang sesuai."""

            # If no ANC visits yet, recommend first visit
            if not anc_visits:
                # Calculate expected pregnancy weeks based on HPHT
                hpht_str = current_pregnancy.get('tanggal_hpht', '')
                if hpht_str:
                    try:
                        hpht_date = datetime.strptime(hpht_str, '%Y-%m-%d')
                        today = datetime.now()
                        days_pregnant = (today - hpht_date).days
                        weeks_pregnant = int(days_pregnant / 7)
                        
                        # Recommend immediate visit if more than 8 weeks
                        if weeks_pregnant >= 8:
                            next_visit_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
                            return f"""Berdasarkan tanggal HPHT Anda ({hpht_str}), perkiraan usia kehamilan saat ini adalah sekitar {weeks_pregnant} minggu.

Anda belum memiliki riwayat kunjungan ANC. Sangat disarankan untuk segera melakukan kunjungan ANC pertama.

📅 **Jadwal yang disarankan: {next_visit_date} (besok)**

Kunjungan ANC pertama sebaiknya dilakukan pada usia kehamilan 8-12 minggu untuk:
• Pemeriksaan kondisi ibu dan janin
• Skrining risiko kehamilan
• Pemberian suplemen asam folat
• Penjadwalan kunjungan ANC selanjutnya"""
                        else:
                            target_week = 8
                            days_to_wait = (target_week * 7) - days_pregnant
                            next_visit_date = (today + timedelta(days=days_to_wait)).strftime('%Y-%m-%d')
                            return f"""Berdasarkan tanggal HPHT Anda ({hpht_str}), perkiraan usia kehamilan saat ini adalah sekitar {weeks_pregnant} minggu.

📅 **Jadwal kunjungan ANC pertama yang disarankan: {next_visit_date}**
(Pada usia kehamilan 8 minggu)

Kunjungan ANC pertama sebaiknya dilakukan pada usia kehamilan 8-12 minggu."""
                    except ValueError:
                        pass
                
                return """Belum ada riwayat kunjungan ANC yang tercatat untuk kehamilan Anda saat ini.
                
📅 **Rekomendasi: Lakukan kunjungan ANC pertama secepatnya**

Sangat penting untuk melakukan pemeriksaan ANC rutin sesuai jadwal:
• Trimester 1 (0-12 minggu): minimal 1 kali kunjungan
• Trimester 2 (13-28 minggu): setiap 4 minggu sekali  
• Trimester 3 (29-40 minggu): setiap 2 minggu sekali, dan setiap minggu setelah 36 minggu"""

            # Get the most recent ANC visit
            latest_visit = anc_visits[0]  # Already sorted by date descending
            last_visit_date_str = latest_visit.get('tanggal_kunjungan', '')
            last_pregnancy_weeks = latest_visit.get('usia_kehamilan', 0)
            
            if not last_visit_date_str or not last_pregnancy_weeks:
                return "Data kunjungan ANC tidak lengkap. Silakan hubungi bidan untuk informasi jadwal kontrol berikutnya."
            
            # Parse last visit date
            try:
                last_visit_date = datetime.strptime(last_visit_date_str, '%Y-%m-%d')
            except ValueError:
                return "Format tanggal kunjungan tidak valid. Silakan hubungi bidan untuk informasi jadwal kontrol berikutnya."
            
            # Calculate current pregnancy weeks based on days elapsed
            today = datetime.now()
            days_elapsed = (today - last_visit_date).days
            weeks_elapsed = days_elapsed / 7
            current_pregnancy_weeks = int(last_pregnancy_weeks + weeks_elapsed)
            
            # Determine next visit interval based on WHO/Ministry of Health guidelines
            if current_pregnancy_weeks <= 12:
                # Trimester 1: Next visit at week 13 (start of trimester 2)
                target_weeks = 13
                interval_name = "memasuki trimester 2"
            elif current_pregnancy_weeks <= 28:
                # Trimester 2: every 4 weeks
                weeks_since_last = current_pregnancy_weeks - last_pregnancy_weeks
                if weeks_since_last >= 4:
                    # Already overdue
                    target_weeks = current_pregnancy_weeks
                    interval_name = "kontrol trimester 2 (terlambat)"
                else:
                    # Calculate next 4-week interval
                    target_weeks = last_pregnancy_weeks + 4
                    interval_name = "kontrol trimester 2"
            elif current_pregnancy_weeks <= 36:
                # Trimester 3 (29-36 weeks): every 2 weeks
                weeks_since_last = current_pregnancy_weeks - last_pregnancy_weeks
                if weeks_since_last >= 2:
                    # Already overdue
                    target_weeks = current_pregnancy_weeks
                    interval_name = "kontrol trimester 3 (terlambat)"
                else:
                    # Calculate next 2-week interval
                    target_weeks = last_pregnancy_weeks + 2
                    interval_name = "kontrol trimester 3"
            else:
                # After 36 weeks: every week
                weeks_since_last = current_pregnancy_weeks - last_pregnancy_weeks
                if weeks_since_last >= 1:
                    # Already overdue
                    target_weeks = current_pregnancy_weeks
                    interval_name = "kontrol menjelang persalinan (terlambat)"
                else:
                    # Next week
                    target_weeks = last_pregnancy_weeks + 1
                    interval_name = "kontrol menjelang persalinan"
            
            # Calculate exact target date
            if target_weeks <= current_pregnancy_weeks:
                # Overdue or due now - recommend immediate visit
                next_visit_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')  # Tomorrow
                urgency_message = f"**SEGERA - {next_visit_date} (besok)**"
                urgency_note = "⚠️ **PENTING**: Anda sudah melewati jadwal kontrol yang disarankan."
            else:
                # Calculate exact date
                weeks_to_add = target_weeks - current_pregnancy_weeks
                days_to_add = int(weeks_to_add * 7)
                target_date = today + timedelta(days=days_to_add)
                next_visit_date = target_date.strftime('%Y-%m-%d')
                urgency_message = f"**{next_visit_date}**"
                urgency_note = ""
            
            # Format the response message
            response_message = f"""Berdasarkan kunjungan ANC terakhir Anda pada {last_visit_date_str} dengan usia kehamilan {last_pregnancy_weeks} minggu:

📊 **Status Kehamilan Saat Ini:**
• Usia kehamilan: sekitar {current_pregnancy_weeks} minggu
• Status: {current_pregnancy.get('status_kehamilan', 'Berjalan')}

📅 **Jadwal Kontrol ANC Berikutnya:**
• Tanggal: {urgency_message}
• Target usia kehamilan: {target_weeks} minggu
• Jenis kontrol: {interval_name}

{urgency_note}"""
            
            # Add appropriate guidance based on trimester
            if current_pregnancy_weeks <= 12:
                response_message += "\n\n📋 **Trimester 1**: Pemeriksaan penting untuk memantau perkembangan awal janin dan deteksi risiko."
            elif current_pregnancy_weeks <= 28:
                response_message += "\n\n📋 **Trimester 2**: Kontrol setiap 4 minggu untuk memantau pertumbuhan janin dan kesehatan ibu."
            elif current_pregnancy_weeks <= 36:
                response_message += "\n\n📋 **Trimester 3**: Kontrol setiap 2 minggu untuk persiapan persalinan dan pemantauan intensif."
            else:
                response_message += "\n\n📋 **Menjelang Persalinan**: Kontrol setiap minggu untuk memantau tanda-tanda persalinan dan kesiapan ibu."
            
            response_message += "\n\n💡 **Catatan**: Harap hadir sesuai jadwal agar kondisi ibu dan janin tetap terpantau dengan baik. Jika ada keluhan atau gejala tidak normal, segera konsultasikan dengan bidan atau dokter."
            
            return response_message
            
        except Exception as e:
            logger.error(f"Error calculating ANC reminder: {str(e)}")
            return "Terjadi kesalahan dalam menghitung jadwal ANC. Silakan hubungi bidan untuk informasi jadwal kontrol berikutnya."
    
    def generate_recommendations(self, user_history: Dict[str, Any], 
                               customer_name: str) -> List[str]:
        """
        Generate personalized question recommendations based on user history
        
        Args:
            user_history (dict): User's medical history summary
            customer_name (str): Customer's name
            
        Returns:
            List[str]: List of recommended questions
        """
        try:
            if not self.model:
                # Fallback recommendations - SAFE HISTORICAL DATA QUERIES
                return [
                    "Tampilkan ringkasan data kesehatan saya",
                    "Apakah ada catatan kunjungan terakhir?",
                    "Siapa dokter yang biasa menangani saya?",
                    "Golongan darah saya apa?"
                ]
            
            prompt = f"""
Berdasarkan riwayat medis pasien berikut, buatlah 4 pertanyaan rekomendasi yang HANYA fokus pada data historis yang sudah tercatat.

NAMA PASIEN: {customer_name}

RIWAYAT MEDIS:
{self._format_database_context(user_history)}

ATURAN KETAT:
- HANYA buat pertanyaan tentang data historis yang sudah ada
- JANGAN buat pertanyaan yang meminta opini medis atau diagnosis
- JANGAN rekomendasikan pertanyaan konsultasi medis
- Fokus pada kueri data: "Tampilkan...", "Apakah ada catatan...", "Bagaimana tren..."

TUGAS:
Buat 4 pertanyaan yang:
1. Menanyakan data historis yang tercatat (hasil lab, kunjungan, obat, dll.)
2. Bersifat administratif dan faktual saja
3. Aman dari aspek medis (tidak meminta saran atau opini)
4. Menggunakan bahasa Indonesia yang natural

FORMAT JAWABAN:
Berikan hanya 4 pertanyaan, satu per baris, tanpa numbering atau bullet points.

Contoh format yang AMAN:
Tampilkan hasil lab terakhir saya
Apakah ada catatan kunjungan bulan lalu?
Siapa dokter yang biasa menangani saya?
Bagaimana tren berat badan dari data ANC sebelumnya?

Pertanyaan rekomendasi:
"""
            
            response = self.model.generate_content(prompt)
            
            if response and response.text:
                questions = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
                # Filter out empty questions and take first 4
                questions = [q for q in questions if q and len(q) > 10][:4]
                
                if len(questions) >= 4:
                    return questions
                else:
                    # Pad with default questions if needed - SAFE HISTORICAL QUERIES
                    default_questions = [
                        "Tampilkan ringkasan data kesehatan saya",
                        "Apakah ada catatan kunjungan terakhir?", 
                        "Siapa dokter yang biasa menangani saya?",
                        "Golongan darah saya apa?"
                    ]
                    return (questions + default_questions)[:4]
            
            # Fallback if API fails - SAFE HISTORICAL DATA QUERIES
            return [
                "Tampilkan ringkasan data kesehatan saya",
                "Apakah ada catatan kunjungan terakhir?",
                "Siapa dokter yang biasa menangani saya?", 
                "Golongan darah saya apa?"
            ]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            # Return default recommendations on error - SAFE HISTORICAL QUERIES
            return [
                "Tampilkan ringkasan data kesehatan saya",
                "Apakah ada catatan kunjungan terakhir?",
                "Siapa dokter yang biasa menangani saya?",
                "Golongan darah saya apa?"
            ]
