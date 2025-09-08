import pandas as pd
import os
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DatabaseHandler:
    """Handle database operations with CSV files"""
    
    def __init__(self, base_path):
        self.base_path = base_path
        self.db_path = os.path.join(base_path, "Database")
        self.knowledge_base_path = base_path
        
        # Initialize data containers
        self.tables = {}
        
        # Load all CSV files
        self._load_all_tables()
        
        # Load knowledge base files
        self._load_knowledge_base()
    
    def _load_all_tables(self):
        """Load all CSV files from Database directory"""
        try:
            csv_files = [
                'customer.csv',
                'anc_kunjungan.csv',
                'diagnosis.csv',
                'dokter.csv',
                'hasil_lab.csv',
                'historikal_kondisi_fisik.csv',
                'imunisasi_ibu_hamil.csv',
                'jadwal_dokter.csv',
                'kehamilan.csv',
                'persalinan.csv',
                'preskripsi.csv',
                'riwayat_berobat.csv',
                'suplemen_ibu_hamil.csv'
            ]
            
            for csv_file in csv_files:
                file_path = os.path.join(self.db_path, csv_file)
                if os.path.exists(file_path):
                    try:
                        table_name = csv_file.replace('.csv', '')
                        df = pd.read_csv(file_path)
                        # Clean column names
                        df.columns = df.columns.str.strip()
                        self.tables[table_name] = df
                        logger.info(f"Loaded table '{table_name}' with {len(df)} records")
                    except Exception as e:
                        logger.error(f"Error loading {csv_file}: {str(e)}")
                else:
                    logger.warning(f"CSV file not found: {file_path}")
            
            logger.info(f"Successfully loaded {len(self.tables)} database tables")
            
        except Exception as e:
            logger.error(f"Error loading database tables: {str(e)}")
            raise
    
    def _load_knowledge_base(self):
        """Load knowledge base files"""
        try:
            # Load pregnancy guidance knowledge (merged intent)
            knowledge_file = os.path.join(self.knowledge_base_path, "panduan_persiapan_persalinan.txt")
            if os.path.exists(knowledge_file):
                with open(knowledge_file, 'r', encoding='utf-8') as f:
                    self.pregnancy_knowledge = f.read()
                logger.info("Loaded pregnancy guidance knowledge base")
            else:
                self.pregnancy_knowledge = ""
                logger.warning(f"Pregnancy guidance knowledge file not found: {knowledge_file}")
            
            # Load intent descriptions
            intent_desc_file = os.path.join(self.knowledge_base_path, "deskripsi_inten.md")
            if os.path.exists(intent_desc_file):
                with open(intent_desc_file, 'r', encoding='utf-8') as f:
                    self.intent_descriptions = f.read()
                logger.info("Loaded intent descriptions")
            else:
                self.intent_descriptions = ""
                logger.warning(f"Intent descriptions file not found: {intent_desc_file}")
                
        except Exception as e:
            logger.error(f"Error loading knowledge base: {str(e)}")
            self.pregnancy_knowledge = ""
            self.intent_descriptions = ""
    
    def get_context_for_intent(self, intent, customer_id):
        """
        Get relevant database context for a specific intent and customer
        
        Args:
            intent (str): The classified intent
            customer_id (str): Customer ID
            
        Returns:
            dict: Context data from relevant tables
        """
        try:
            context = {
                'intent': intent,
                'customer_id': customer_id,
                'data': {},
                'knowledge_base': {},
                'intent_description': self._get_intent_description(intent)
            }
            
            # Route to specific context getter based on intent
            if intent == 'reminder_kontrol_kehamilan':
                context['data'] = self._get_anc_schedule_context(customer_id)
                
            elif intent == 'anc_tracker':
                context['data'] = self._get_anc_tracking_context(customer_id)
                
            elif intent == 'imunisasi_tracker':
                context['data'] = self._get_immunization_context(customer_id)
                
            elif intent == 'riwayat_persalinan':
                context['data'] = self._get_delivery_history_context(customer_id)
                
            elif intent == 'riwayat_suplemen_kehamilan':
                context['data'] = self._get_supplement_context(customer_id)
                
            elif intent == 'riwayat_kondisi_fisik':
                context['data'] = self._get_physical_condition_context(customer_id)
                
            elif intent == 'cek_golongan_darah':
                context['data'] = self._get_blood_type_context(customer_id)
                
            elif intent == 'cek_data_customer':
                context['data'] = self._get_customer_data_context(customer_id)
                
            elif intent in ['riwayat_diagnosis', 'detail_diagnosis']:
                context['data'] = self._get_diagnosis_context(customer_id)
                
            elif intent in ['riwayat_preskripsi_obat', 'detail_preskripsi_obat']:
                context['data'] = self._get_prescription_context(customer_id)
                
            elif intent == 'riwayat_berobat':
                context['data'] = self._get_treatment_history_context(customer_id)
                
            elif intent == 'jadwal_dokter':
                context['data'] = self._get_doctor_schedule_context(customer_id)
                
            elif intent == 'detail_dokter':
                context['data'] = self._get_doctor_details_context(customer_id)
                
            elif intent in ['hasil_lab_ringkasan', 'hasil_lab_detail']:
                context['data'] = self._get_lab_results_context(customer_id)
                
            elif intent == 'panduan_persiapan_persalinan':
                # Updated: Now handles both pregnancy guidance and general questions
                context['knowledge_base']['pregnancy_info'] = self.pregnancy_knowledge
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting context for intent {intent}: {str(e)}")
            return {'intent': intent, 'customer_id': customer_id, 'data': {}, 'knowledge_base': {}}
    
    def _get_intent_description(self, intent):
        """Get description for specific intent from markdown file"""
        try:
            lines = self.intent_descriptions.split('\n')
            for line in lines:
                if intent in line and '|' in line:
                    parts = line.split('|')
                    if len(parts) > 2:
                        return parts[2].strip()
            return ""
        except Exception as e:
            logger.error(f"Error getting intent description: {str(e)}")
            return ""
    
    def _get_anc_schedule_context(self, customer_id):
        """Get ANC schedule context"""
        context = {}
        
        # First get pregnancy ID from customer_id
        pregnancy_ids = []
        if 'kehamilan' in self.tables:
            pregnancy_data = self.tables['kehamilan'][
                self.tables['kehamilan']['customer_id'] == customer_id
            ]
            if not pregnancy_data.empty:
                pregnancy_ids = pregnancy_data['id_kehamilan'].tolist()
                context['pregnancy_data'] = pregnancy_data.to_dict('records')
        
        # Get ANC visits using pregnancy IDs
        if 'anc_kunjungan' in self.tables and pregnancy_ids:
            anc_visits = self.tables['anc_kunjungan'][
                self.tables['anc_kunjungan']['id_kehamilan'].isin(pregnancy_ids)
            ].sort_values('tanggal_kunjungan', ascending=False)
            context['anc_visits'] = anc_visits.to_dict('records')
        else:
            context['anc_visits'] = []
        
        # Get delivery history to check if pregnancy is still ongoing
        if 'persalinan' in self.tables and pregnancy_ids:
            deliveries = self.tables['persalinan'][
                self.tables['persalinan']['id_kehamilan'].isin(pregnancy_ids)
            ].sort_values('tanggal_lahir', ascending=False)
            context['deliveries'] = deliveries.to_dict('records')
        else:
            context['deliveries'] = []
        
        return context
    
    def _get_anc_tracking_context(self, customer_id):
        """Get ANC tracking context"""
        context = {}
        
        # First get pregnancy ID from customer_id
        pregnancy_ids = []
        if 'kehamilan' in self.tables:
            pregnancy_data = self.tables['kehamilan'][
                self.tables['kehamilan']['customer_id'] == customer_id
            ]
            if not pregnancy_data.empty:
                pregnancy_ids = pregnancy_data['id_kehamilan'].tolist()
                context['pregnancy_data'] = pregnancy_data.to_dict('records')
        
        # Get ANC visits using pregnancy IDs
        if 'anc_kunjungan' in self.tables and pregnancy_ids:
            anc_visits = self.tables['anc_kunjungan'][
                self.tables['anc_kunjungan']['id_kehamilan'].isin(pregnancy_ids)
            ].sort_values('tanggal_kunjungan', ascending=False)
            context['anc_visits'] = anc_visits.to_dict('records')
        else:
            context['anc_visits'] = []
        
        return context
    
    def _get_immunization_context(self, customer_id):
        """Get immunization context"""
        context = {}
        
        # First get pregnancy ID from customer_id
        pregnancy_ids = []
        if 'kehamilan' in self.tables:
            pregnancy_data = self.tables['kehamilan'][
                self.tables['kehamilan']['customer_id'] == customer_id
            ]
            if not pregnancy_data.empty:
                pregnancy_ids = pregnancy_data['id_kehamilan'].tolist()
        
        # Get immunizations using pregnancy IDs
        if 'imunisasi_ibu_hamil' in self.tables and pregnancy_ids:
            immunizations = self.tables['imunisasi_ibu_hamil'][
                self.tables['imunisasi_ibu_hamil']['id_kehamilan'].isin(pregnancy_ids)
            ].sort_values('tanggal_pemberian', ascending=False)
            context['immunizations'] = immunizations.to_dict('records')
        else:
            context['immunizations'] = []
        
        return context
    
    def _get_delivery_history_context(self, customer_id):
        """Get delivery history context"""
        context = {}
        
        # First get pregnancy IDs from customer_id
        pregnancy_ids = []
        if 'kehamilan' in self.tables:
            pregnancy_data = self.tables['kehamilan'][
                self.tables['kehamilan']['customer_id'] == customer_id
            ]
            if not pregnancy_data.empty:
                pregnancy_ids = pregnancy_data['id_kehamilan'].tolist()
                context['pregnancies'] = pregnancy_data.to_dict('records')
        
        # Get deliveries using pregnancy IDs
        if 'persalinan' in self.tables and pregnancy_ids:
            deliveries = self.tables['persalinan'][
                self.tables['persalinan']['id_kehamilan'].isin(pregnancy_ids)
            ].sort_values('tanggal_lahir', ascending=False)
            context['deliveries'] = deliveries.to_dict('records')
        else:
            context['deliveries'] = []
        
        return context
    
    def _get_supplement_context(self, customer_id):
        """Get supplement context"""
        context = {}
        
        # First get pregnancy IDs from customer_id
        pregnancy_ids = []
        anc_visit_ids = []
        
        if 'kehamilan' in self.tables:
            pregnancy_data = self.tables['kehamilan'][
                self.tables['kehamilan']['customer_id'] == customer_id
            ]
            if not pregnancy_data.empty:
                pregnancy_ids = pregnancy_data['id_kehamilan'].tolist()
        
        # Get ANC visit IDs for this customer
        if 'anc_kunjungan' in self.tables and pregnancy_ids:
            anc_visits = self.tables['anc_kunjungan'][
                self.tables['anc_kunjungan']['id_kehamilan'].isin(pregnancy_ids)
            ]
            anc_visit_ids = anc_visits['id_kunjungan'].tolist()
        
        # Get supplements using ANC visit IDs
        if 'suplemen_ibu_hamil' in self.tables and anc_visit_ids:
            supplements = self.tables['suplemen_ibu_hamil'][
                self.tables['suplemen_ibu_hamil']['id_kunjungan'].isin(anc_visit_ids)
            ]
            # Sort by the date from related ANC visit
            if 'anc_kunjungan' in self.tables:
                # Merge with ANC data to get dates for sorting
                supplements_with_dates = supplements.merge(
                    self.tables['anc_kunjungan'][['id_kunjungan', 'tanggal_kunjungan']], 
                    on='id_kunjungan', 
                    how='left'
                ).sort_values('tanggal_kunjungan', ascending=False)
                context['supplements'] = supplements_with_dates.to_dict('records')
            else:
                context['supplements'] = supplements.to_dict('records')
        else:
            context['supplements'] = []
        
        return context
    
    def _get_physical_condition_context(self, customer_id):
        """Get physical condition context"""
        context = {}
        
        if 'historikal_kondisi_fisik' in self.tables:
            conditions = self.tables['historikal_kondisi_fisik'][
                self.tables['historikal_kondisi_fisik']['customer_id'] == customer_id
            ].sort_values('tanggal_pemeriksaan', ascending=False)
            context['physical_conditions'] = conditions.to_dict('records')
        
        return context
    
    def _get_blood_type_context(self, customer_id):
        """Get blood type context"""
        context = {}
        
        if 'customer' in self.tables:
            customer = self.tables['customer'][
                self.tables['customer']['customer_id'] == customer_id
            ]
            if not customer.empty:
                context['customer_info'] = customer.iloc[0].to_dict()
        
        return context
    
    def _get_customer_data_context(self, customer_id):
        """Get customer data context (personal information)"""
        context = {}
        
        if 'customer' in self.tables:
            customer = self.tables['customer'][
                self.tables['customer']['customer_id'] == customer_id
            ]
            if not customer.empty:
                customer_data = customer.iloc[0].to_dict()
                context['customer_info'] = customer_data
                
                # Add specific fields for easy access
                context['name'] = customer_data.get('name', 'Tidak diketahui')
                context['alamat'] = customer_data.get('alamat', 'Tidak tersedia')
                context['no_hp'] = customer_data.get('no_hp', 'Tidak tersedia')
                context['email'] = customer_data.get('email', 'Tidak tersedia')
                context['tanggal_lahir'] = customer_data.get('tanggal_lahir', 'Tidak tersedia')
                context['golongan_darah'] = customer_data.get('golongan_darah', 'Tidak tersedia')
                context['NIK'] = customer_data.get('NIK', 'Tidak tersedia')
        
        return context
    
    def _get_diagnosis_context(self, customer_id):
        """Get diagnosis context"""
        context = {}
        
        if 'diagnosis' in self.tables:
            # Get visits for this customer first
            if 'riwayat_berobat' in self.tables:
                visits = self.tables['riwayat_berobat'][
                    self.tables['riwayat_berobat']['customer_id'] == customer_id
                ]
                visit_ids = visits['visit_id'].tolist()
                
                # Get diagnoses for these visits
                diagnoses = self.tables['diagnosis'][
                    self.tables['diagnosis']['visit_id'].isin(visit_ids)
                ].sort_values('created_at', ascending=False)
                
                context['diagnoses'] = diagnoses.to_dict('records')
        
        return context
    
    def _get_prescription_context(self, customer_id):
        """Get prescription context"""
        context = {}
        
        if 'preskripsi' in self.tables:
            # Get visits for this customer first
            if 'riwayat_berobat' in self.tables:
                visits = self.tables['riwayat_berobat'][
                    self.tables['riwayat_berobat']['customer_id'] == customer_id
                ]
                visit_ids = visits['visit_id'].tolist()
                
                # Get prescriptions for these visits
                prescriptions = self.tables['preskripsi'][
                    self.tables['preskripsi']['visit_id'].isin(visit_ids)
                ].sort_values('start_date', ascending=False)
                
                context['prescriptions'] = prescriptions.to_dict('records')
        
        return context
    
    def _get_treatment_history_context(self, customer_id):
        """Get treatment history context"""
        context = {}
        
        if 'riwayat_berobat' in self.tables:
            treatments = self.tables['riwayat_berobat'][
                self.tables['riwayat_berobat']['customer_id'] == customer_id
            ].sort_values('visit_date', ascending=False)  # Fixed column name
            context['treatments'] = treatments.to_dict('records')
        
        return context
    
    def _get_doctor_schedule_context(self, customer_id):
        """Get doctor schedule context"""
        context = {}
        
        if 'jadwal_dokter' in self.tables:
            # Get all doctor schedules (could be filtered by customer's preferred doctors)
            schedules = self.tables['jadwal_dokter'].sort_values('practice_date', ascending=False)
            context['doctor_schedules'] = schedules.to_dict('records')
        
        if 'dokter' in self.tables:
            doctors = self.tables['dokter']
            context['doctors'] = doctors.to_dict('records')
        
        return context
    
    def _get_doctor_details_context(self, customer_id):
        """Get doctor details context"""
        context = {}
        
        if 'dokter' in self.tables:
            doctors = self.tables['dokter']
            context['doctors'] = doctors.to_dict('records')
        
        return context
    
    def _get_lab_results_context(self, customer_id):
        """Get lab results context"""
        context = {}
        
        if 'hasil_lab' in self.tables:
            lab_results = self.tables['hasil_lab'][
                self.tables['hasil_lab']['customer_id'] == customer_id
            ].sort_values('test_date', ascending=False)
            context['lab_results'] = lab_results.to_dict('records')
        
        return context
    
    def get_user_history_summary(self, customer_id):
        """
        Get a comprehensive summary of user's medical history for recommendations
        
        Args:
            customer_id (str): Customer ID
            
        Returns:
            dict: Summary of user's medical history
        """
        try:
            summary = {
                'customer_id': customer_id,
                'recent_visits': [],
                'recent_diagnoses': [],
                'recent_prescriptions': [],
                'recent_lab_results': [],
                'upcoming_appointments': []
            }
            
            # Recent visits
            if 'riwayat_berobat' in self.tables:
                recent_visits = self.tables['riwayat_berobat'][
                    self.tables['riwayat_berobat']['customer_id'] == customer_id
                ].sort_values('visit_date', ascending=False).head(3)  # Fixed column name
                summary['recent_visits'] = recent_visits.to_dict('records')
            
            # Recent diagnoses
            if 'diagnosis' in self.tables and 'riwayat_berobat' in self.tables:
                visits = self.tables['riwayat_berobat'][
                    self.tables['riwayat_berobat']['customer_id'] == customer_id
                ]
                visit_ids = visits['visit_id'].tolist()
                
                recent_diagnoses = self.tables['diagnosis'][
                    self.tables['diagnosis']['visit_id'].isin(visit_ids)
                ].sort_values('created_at', ascending=False).head(3)
                summary['recent_diagnoses'] = recent_diagnoses.to_dict('records')
            
            # Recent prescriptions  
            if 'preskripsi' in self.tables and 'riwayat_berobat' in self.tables:
                visits = self.tables['riwayat_berobat'][
                    self.tables['riwayat_berobat']['customer_id'] == customer_id
                ]
                visit_ids = visits['visit_id'].tolist()
                
                recent_prescriptions = self.tables['preskripsi'][
                    self.tables['preskripsi']['visit_id'].isin(visit_ids)
                ].sort_values('start_date', ascending=False).head(3)
                summary['recent_prescriptions'] = recent_prescriptions.to_dict('records')
            
            # Recent lab results
            if 'hasil_lab' in self.tables:
                recent_lab = self.tables['hasil_lab'][
                    self.tables['hasil_lab']['customer_id'] == customer_id
                ].sort_values('test_date', ascending=False).head(3)
                summary['recent_lab_results'] = recent_lab.to_dict('records')
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting user history summary: {str(e)}")
            return {'customer_id': customer_id}
