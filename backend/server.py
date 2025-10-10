from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
import uuid
from datetime import datetime, timezone
import shutil
import cv2
import numpy as np
from PIL import Image
import io
import random

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(title="Sistem Klasifikasi Ikan Betta", version="1.0.0")

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Create uploads directory
UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Define Models
class BettaSpecies(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    nama_umum: str
    nama_ilmiah: str
    deskripsi: str
    karakteristik: List[str]
    habitat: str
    ukuran_avg: str
    gambar_contoh: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ClassificationResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    nama_ikan: str
    tingkat_keyakinan: float
    gambar_path: str
    thumbnail_path: str
    species_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class ClassificationResponse(BaseModel):
    hasil_klasifikasi: str
    tingkat_keyakinan: float
    species_id: Optional[str] = None
    gambar_url: str
    thumbnail_url: str
    riwayat_id: str

class SpeciesCreate(BaseModel):
    nama_umum: str
    nama_ilmiah: str
    deskripsi: str
    karakteristik: List[str]
    habitat: str
    ukuran_avg: str
    gambar_contoh: str

# Mock CNN Model Class
class MockBettaCNN:
    def __init__(self):
        self.betta_types = [
            "Crown Tail", "Half Moon", "Plakat", "Double Tail", 
            "Veiltail", "Delta Tail", "Spade Tail", "Rose Tail",
            "Elephant Ear", "Combtail", "Round Tail", "Super Delta"
        ]
        
    def preprocess_image(self, image_bytes):
        """Simulate image preprocessing like a real CNN"""
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to standard CNN input size
        image = image.resize((224, 224))
        
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        return img_array
    
    def predict(self, image_array):
        """Mock prediction with realistic confidence scores"""
        # Simulate CNN prediction with weighted random selection
        weights = [0.25, 0.20, 0.15, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.02, 0.005, 0.005]
        # Normalize weights to ensure they sum to 1.0
        weights = np.array(weights)
        weights = weights / weights.sum()
        selected_idx = np.random.choice(len(self.betta_types), p=weights)
        
        # Generate realistic confidence score (higher for common types)
        if selected_idx < 3:  # Crown Tail, Half Moon, Plakat
            confidence = random.uniform(0.75, 0.95)
        elif selected_idx < 6:  # Common varieties
            confidence = random.uniform(0.60, 0.85)
        else:  # Rare varieties
            confidence = random.uniform(0.45, 0.75)
            
        return self.betta_types[selected_idx], confidence

# Initialize mock model
mock_cnn = MockBettaCNN()

# Helper Functions
def create_thumbnail(image_path: Path, size=(150, 150)):
    """Create thumbnail for uploaded image"""
    thumb_path = image_path.parent / f"thumb_{image_path.name}"
    
    with Image.open(image_path) as img:
        img.thumbnail(size, Image.Resampling.LANCZOS)
        img.save(thumb_path, "JPEG", quality=85)
    
    return thumb_path

async def init_sample_data():
    """Initialize sample Betta fish data"""
    existing = await db.betta_species.count_documents({})
    if existing > 0:
        return
    
    sample_species = [
        {
            "nama_umum": "Crown Tail",
            "nama_ilmiah": "Betta splendens var. crown tail",
            "deskripsi": "Ikan Betta dengan sirip ekor yang menyerupai mahkota dengan ray yang memanjang",
            "karakteristik": ["Sirip ekor bercabang", "Ray sirip memanjang", "Bentuk seperti mahkota"],
            "habitat": "Air tawar tropis",
            "ukuran_avg": "6-8 cm",
            "gambar_contoh": "https://images.unsplash.com/photo-1544551763-46a013bb70d5?w=300&h=200&fit=crop"
        },
        {
            "nama_umum": "Half Moon",
            "nama_ilmiah": "Betta splendens var. half moon",
            "deskripsi": "Ikan Betta dengan sirip ekor membentuk setengah lingkaran sempurna 180 derajat",
            "karakteristik": ["Sirip ekor membulat", "Bentuk setengah bulan", "Sirip lebar"],
            "habitat": "Air tawar tropis",
            "ukuran_avg": "6-7 cm",
            "gambar_contoh": "https://images.unsplash.com/photo-1520637836862-4d197d17c17a?w=300&h=200&fit=crop"
        },
        {
            "nama_umum": "Plakat",
            "nama_ilmiah": "Betta splendens var. plakat",
            "deskripsi": "Ikan Betta dengan sirip pendek, lebih aktif dan agresif, sering digunakan untuk adu",
            "karakteristik": ["Sirip pendek", "Tubuh kekar", "Sangat aktif"],
            "habitat": "Air tawar tropis",
            "ukuran_avg": "5-6 cm",
            "gambar_contoh": "https://images.unsplash.com/photo-1501472312651-726ebb35d936?w=300&h=200&fit=crop"
        },
        {
            "nama_umum": "Double Tail",
            "nama_ilmiah": "Betta splendens var. double tail",
            "deskripsi": "Ikan Betta dengan sirip ekor yang terbelah menjadi dua bagian",
            "karakteristik": ["Sirip ekor terbelah", "Sirip dorsal lebar", "Bentuk unik"],
            "habitat": "Air tawar tropis",
            "ukuran_avg": "5-7 cm",
            "gambar_contoh": "https://images.unsplash.com/photo-1506561829881-82ef6b43b2a4?w=300&h=200&fit=crop"
        },
        {
            "nama_umum": "Veiltail",
            "nama_ilmiah": "Betta splendens var. veiltail",
            "deskripsi": "Ikan Betta dengan sirip ekor panjang yang menjuntai seperti kerudung",
            "karakteristik": ["Sirip ekor sangat panjang", "Menjuntai ke bawah", "Elegan"],
            "habitat": "Air tawar tropis",
            "ukuran_avg": "6-8 cm",
            "gambar_contoh": "https://images.unsplash.com/photo-1491604612772-6853927639ef?w=300&h=200&fit=crop"
        }
    ]
    
    for species_data in sample_species:
        species = BettaSpecies(**species_data)
        doc = species.model_dump()
        doc['created_at'] = doc['created_at'].isoformat()
        await db.betta_species.insert_one(doc)

# API Routes
@api_router.get("/")
async def root():
    return {"message": "Sistem Klasifikasi Ikan Betta API", "status": "aktif"}

@api_router.get("/species", response_model=List[BettaSpecies])
async def get_all_species():
    """Dapatkan semua jenis ikan Betta"""
    species_list = await db.betta_species.find({}, {"_id": 0}).to_list(1000)
    
    for species in species_list:
        if isinstance(species['created_at'], str):
            species['created_at'] = datetime.fromisoformat(species['created_at'])
    
    return species_list

@api_router.get("/species/{species_id}", response_model=BettaSpecies)
async def get_species_detail(species_id: str):
    """Dapatkan detail spesies berdasarkan ID"""
    species = await db.betta_species.find_one({"id": species_id}, {"_id": 0})
    if not species:
        raise HTTPException(status_code=404, detail="Spesies tidak ditemukan")
    
    if isinstance(species['created_at'], str):
        species['created_at'] = datetime.fromisoformat(species['created_at'])
    
    return species

@api_router.post("/classify", response_model=ClassificationResponse)
async def classify_fish(file: UploadFile = File(...)):
    """Klasifikasi gambar ikan Betta"""
    # Validasi file
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar")
    
    # Check file size (5MB limit)
    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:  # 5MB
        raise HTTPException(status_code=400, detail="Ukuran file maksimal 5MB")
    
    # Save uploaded file
    file_extension = Path(file.filename).suffix
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    file_path = UPLOAD_DIR / unique_filename
    
    with open(file_path, "wb") as buffer:
        buffer.write(contents)
    
    # Create thumbnail
    thumb_path = create_thumbnail(file_path)
    
    # Preprocess and classify
    try:
        img_array = mock_cnn.preprocess_image(contents)
        predicted_type, confidence = mock_cnn.predict(img_array)
        
        # Find matching species in database
        species = await db.betta_species.find_one({"nama_umum": predicted_type}, {"_id": 0})
        species_id = species['id'] if species else None
        
        # Save classification result
        classification = ClassificationResult(
            nama_ikan=predicted_type,
            tingkat_keyakinan=confidence,
            gambar_path=str(file_path),
            thumbnail_path=str(thumb_path),
            species_id=species_id
        )
        
        # Save to database
        doc = classification.model_dump()
        doc['created_at'] = doc['created_at'].isoformat()
        await db.classifications.insert_one(doc)
        
        # Return response
        return ClassificationResponse(
            hasil_klasifikasi=predicted_type,
            tingkat_keyakinan=round(confidence, 2),
            species_id=species_id,
            gambar_url=f"/uploads/{unique_filename}",
            thumbnail_url=f"/uploads/thumb_{unique_filename}",
            riwayat_id=classification.id
        )
        
    except Exception as e:
        # Clean up files on error
        if file_path.exists():
            file_path.unlink()
        if 'thumb_path' in locals() and thumb_path.exists():
            thumb_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error dalam klasifikasi: {str(e)}")

@api_router.get("/history", response_model=List[ClassificationResult])
async def get_classification_history():
    """Dapatkan riwayat klasifikasi"""
    history = await db.classifications.find({}, {"_id": 0}).sort("created_at", -1).to_list(100)
    
    for item in history:
        if isinstance(item['created_at'], str):
            item['created_at'] = datetime.fromisoformat(item['created_at'])
    
    return history

@api_router.delete("/history/{classification_id}")
async def delete_classification(classification_id: str):
    """Hapus riwayat klasifikasi"""
    result = await db.classifications.delete_one({"id": classification_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Riwayat tidak ditemukan")
    
    return {"message": "Riwayat berhasil dihapus"}

@api_router.post("/species", response_model=BettaSpecies)
async def create_species(species_data: SpeciesCreate):
    """Tambah spesies baru (admin)"""
    species = BettaSpecies(**species_data.model_dump())
    
    doc = species.model_dump()
    doc['created_at'] = doc['created_at'].isoformat()
    await db.betta_species.insert_one(doc)
    
    return species

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    logger.info("Inisialisasi Sistem Klasifikasi Ikan Betta")
    await init_sample_data()
    logger.info("Data sampel Betta berhasil dimuat")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()