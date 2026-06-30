import os
import io
import datetime
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
from PIL import Image
import pdfplumber
import fitz  # PyMuPDF
import json
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Configuración de dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
PDF_FOLDER = "./pdfs"
OCR_OUTPUT_FOLDER = "./ocr_outputs"
IMAGE_OUTPUT_FOLDER = "./images"
MODEL_OUTPUT_FOLDER = "./outputModel"

# Crear directorios necesarios
for folder in [OCR_OUTPUT_FOLDER, IMAGE_OUTPUT_FOLDER, MODEL_OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Pretrained models
bert_tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT")
bert_model = AutoModel.from_pretrained("tbs17/MathBERT").to(device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Transforms for CLIP image preprocessing
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711]),
])


# --- Procesamiento de PDFs ---
def extract_text_and_images_with_math(pdf_path, ocr_folder, image_folder):
    """
    Extrae texto, contenido matemático y gráficos de un PDF con manejo de errores.
    """
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"Procesando PDF: {pdf_name}")

    try:
        # Text extraction with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text()
                    if text:
                        text_path = os.path.join(ocr_folder, f"{pdf_name}_page{page_idx}.txt")
                        with open(text_path, "w", encoding='utf-8') as text_file:
                            text_file.write(text)
                except Exception as e:
                    print(f"Error al extraer texto de la página {page_idx}: {str(e)}")

        # Image extraction with PyMuPDF
        doc = fitz.open(pdf_path)
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            try:
                for img_index, img in enumerate(page.get_images(full=True)):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        
                        if base_image:
                            image_bytes = base_image["image"]
                            image_ext = base_image["ext"]
                            
                            # Crear un archivo temporal con la extensión correcta
                            temp_path = os.path.join(image_folder, f"{pdf_name}_page{page_idx}_img{img_index}_temp.{image_ext}")
                            with open(temp_path, "wb") as temp_file:
                                temp_file.write(image_bytes)
                            
                            # Abrir y convertir la imagen
                            with Image.open(temp_path) as image:
                                image = image.convert("RGB")
                                final_path = os.path.join(image_folder, f"{pdf_name}_page{page_idx}_img{img_index}.png")
                                image.save(final_path, "PNG")
                            
                            # Eliminar archivo temporal
                            os.remove(temp_path)
                            print(f"Imagen guardada: {final_path}")
                            
                    except Exception as e:
                        print(f"Error al procesar imagen {img_index} en página {page_idx}: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Error al procesar imágenes en página {page_idx}: {str(e)}")
                continue
                
        doc.close()
        
    except Exception as e:
        print(f"Error al procesar el PDF {pdf_path}: {str(e)}")
        raise


# Procesar todos los PDFs
print(f"Buscando PDFs en: {PDF_FOLDER}")
pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
print(f"PDFs encontrados: {len(pdf_files)}")

for pdf_file in pdf_files:
    try:
        print(f"\nProcesando: {pdf_file}")
        extract_text_and_images_with_math(
            os.path.join(PDF_FOLDER, pdf_file),
            OCR_OUTPUT_FOLDER,
            IMAGE_OUTPUT_FOLDER,
        )
        print(f"Procesamiento completado: {pdf_file}")
    except Exception as e:
        print(f"Error al procesar {pdf_file}: {str(e)}")
        print("Continuando con el siguiente PDF...")
        continue

print("\nProcesamiento de PDFs completado")

# --- Dataset ---
def collate_fn(batch):
    """
    Función personalizada para procesar los lotes de datos.
    """
    text_inputs_list = []
    image_inputs_list = []
    
    # Procesar cada elemento del batch
    for text_input, image_input in batch:
        # Asegurar que los tensores de texto tengan la forma correcta
        if 'input_ids' in text_input and 'attention_mask' in text_input:
            # Asegurar que los tensores sean 2D [batch_size, sequence_length]
            input_ids = text_input['input_ids']
            attention_mask = text_input['attention_mask']
            
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
                
            text_input = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
            
        text_inputs_list.append(text_input)
        
        # Asegurar que los tensores de imagen tengan la forma correcta
        if 'pixel_values' in image_input:
            pixel_values = image_input['pixel_values']
            if pixel_values.dim() == 3:  # Si falta la dimensión del batch
                pixel_values = pixel_values.unsqueeze(0)
            image_input = {'pixel_values': pixel_values}
            
        image_inputs_list.append(image_input)
    
    try:
        # Combinar los inputs
        combined_text_inputs = {
            'input_ids': torch.cat([x['input_ids'] for x in text_inputs_list], dim=0),
            'attention_mask': torch.cat([x['attention_mask'] for x in text_inputs_list], dim=0)
        }
        
        combined_image_inputs = {
            'pixel_values': torch.cat([x['pixel_values'] for x in image_inputs_list], dim=0)
        }
        
        return combined_text_inputs, combined_image_inputs
    except Exception as e:
        print("Error al combinar tensores en collate_fn:")
        print(f"Tamaños de texto: {[x['input_ids'].shape for x in text_inputs_list]}")
        print(f"Tamaños de imagen: {[x['pixel_values'].shape for x in image_inputs_list]}")
        raise e

class MultimodalDataset(Dataset):
    def __init__(self, text_files, image_files):
        assert len(text_files) == len(image_files), "Number of text and image files must match"
        self.text_files = text_files
        self.image_files = image_files
        print("Inicializando dataset...")
        print(f"Archivos de texto: {len(text_files)}")
        print(f"Archivos de imagen: {len(image_files)}")

    def __len__(self):
        return len(self.text_files)

    def __getitem__(self, idx):
        try:
            # Cargar y procesar texto
            with open(self.text_files[idx], "r", encoding='utf-8') as file:
                text = file.read().strip()
            
            # Tokenizar texto sin mover a GPU todavía
            text_inputs = bert_tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Asegurar que los tensores de texto tengan la forma correcta [1, sequence_length]
            if text_inputs['input_ids'].dim() == 1:
                text_inputs = {
                    'input_ids': text_inputs['input_ids'].unsqueeze(0),
                    'attention_mask': text_inputs['attention_mask'].unsqueeze(0)
                }
            
            # Cargar y procesar imagen
            image_path = self.image_files[idx]
            image = Image.open(image_path).convert("RGB")
            image_inputs = clip_processor(images=image, return_tensors="pt")
            
            # Verificar y registrar las dimensiones
            if idx == 0:  # Solo para el primer item
                print(f"\nDimensiones de los tensores:")
                print(f"Text input_ids: {text_inputs['input_ids'].shape}")
                print(f"Text attention_mask: {text_inputs['attention_mask'].shape}")
                print(f"Image pixel_values: {image_inputs['pixel_values'].shape}")
            
            return text_inputs, image_inputs
            
        except Exception as e:
            print(f"Error procesando item {idx}: {str(e)}")
            print(f"Archivo de texto: {self.text_files[idx]}")
            print(f"Archivo de imagen: {self.image_files[idx]}")
            raise


# Datos
print("\nPreparando datos para entrenamiento...")
text_files = sorted([os.path.join(OCR_OUTPUT_FOLDER, f) for f in os.listdir(OCR_OUTPUT_FOLDER) if f.endswith(".txt")])
image_files = sorted([os.path.join(IMAGE_OUTPUT_FOLDER, f) for f in os.listdir(IMAGE_OUTPUT_FOLDER) if f.endswith(".png")])

print(f"Archivos de texto encontrados: {len(text_files)}")
print(f"Archivos de imagen encontrados: {len(image_files)}")

# Emparejar archivos de texto e imagen
paired_files = []
for text_file in text_files:
    base_name = os.path.splitext(os.path.basename(text_file))[0]
    # Buscar imágenes correspondientes
    matching_images = [img for img in image_files if os.path.splitext(os.path.basename(img))[0].startswith(base_name)]
    if matching_images:
        paired_files.append((text_file, matching_images[0]))

if not paired_files:
    raise ValueError("No se encontraron pares de archivos texto-imagen coincidentes")

print(f"Pares texto-imagen encontrados: {len(paired_files)}")

# Separar en texto e imágenes emparejados
text_files_paired, image_files_paired = zip(*paired_files)

# Split en train/test
train_texts, test_texts, train_images, test_images = train_test_split(
    list(text_files_paired), 
    list(image_files_paired), 
    test_size=0.2, 
    random_state=42
)

# Crear datasets y dataloaders
print("\nCreando datasets...")
train_dataset = MultimodalDataset(train_texts, train_images)
test_dataset = MultimodalDataset(test_texts, test_images)

print("\nCreando dataloaders...")
train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,  # Ajustar según necesidad
    pin_memory=True if torch.cuda.is_available() else False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,  # Ajustar según necesidad
    pin_memory=True if torch.cuda.is_available() else False
)


# --- Modelo ---
class MultimodalAttentionModel(nn.Module):
    def __init__(self, text_hidden_size=768, image_hidden_size=512, combined_hidden_size=256):
        super(MultimodalAttentionModel, self).__init__()
        self.text_fc = nn.Linear(text_hidden_size, combined_hidden_size)
        self.image_fc = nn.Linear(image_hidden_size, combined_hidden_size)
        self.attention = nn.MultiheadAttention(embed_dim=combined_hidden_size, num_heads=4)
        self.classifier = nn.Linear(combined_hidden_size, 10)

    def forward(self, text_features, image_features):
        text_out = self.text_fc(text_features)
        image_out = self.image_fc(image_features)
        attention_out, _ = self.attention(text_out.unsqueeze(1), image_out.unsqueeze(1), image_out.unsqueeze(1))
        combined = attention_out.squeeze(1)
        return self.classifier(combined)


model = MultimodalAttentionModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# --- Entrenamiento ---
def train_model(model, train_loader, epochs=5, save_path="multimodal_model.pth", save_interval=1):
    """
    Entrena el modelo con checkpoints periódicos.
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        epochs: Número total de épocas
        save_path: Ruta base para guardar el modelo
        save_interval: Cada cuántas épocas guardar un checkpoint
    """
    model.train()
    best_loss = float('inf')
    
    try:
        for epoch in range(epochs):
            epoch_loss = 0
            batch_count = 0
            
            for text_inputs, image_inputs in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                try:
                    # Mover datos a GPU si está disponible
                    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                    image_inputs = {k: v.to(device) for k, v in image_inputs.items()}
                    
                    # Obtener embeddings
                    text_outputs = bert_model(**text_inputs).pooler_output
                    image_outputs = clip_model.get_image_features(**image_inputs)
                    
                    # Generar etiquetas aleatorias del tamaño correcto del batch
                    batch_size = text_outputs.size(0)
                    labels = torch.randint(0, 10, (batch_size,)).to(device)
                    
                    # Forward pass
                    outputs = model(text_outputs, image_outputs)
                    
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    batch_count += 1
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("WARNING: out of memory")
                        if hasattr(torch.cuda, 'empty_cache'):
                            torch.cuda.empty_cache()
                    else:
                        raise e
            
            avg_epoch_loss = epoch_loss / batch_count
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
            
            # Guardar checkpoint si es necesario
            if (epoch + 1) % save_interval == 0:
                checkpoint_path = f"{save_path.replace('.pth', f'_epoch_{epoch+1}.pth')}"
                save_checkpoint(model, optimizer, epoch + 1, avg_epoch_loss, checkpoint_path)
            
            # Guardar mejor modelo
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                best_model_path = save_path.replace('.pth', '_best.pth')
                save_checkpoint(model, optimizer, epoch + 1, best_loss, best_model_path)
                
    except KeyboardInterrupt:
        print("\nEntrenamiento interrumpido. Guardando último checkpoint...")
        save_checkpoint(model, optimizer, epoch + 1, epoch_loss, save_path.replace('.pth', '_interrupted.pth'))
        
    # Guardar modelo final
    final_save_path = save_path.replace('.pth', '_final.pth')
    save_checkpoint(model, optimizer, epochs, epoch_loss, final_save_path)

def save_checkpoint(model, optimizer, epoch, loss, path="multimodal_model.pth"):
    """
    Guarda un checkpoint del modelo con toda la información necesaria para continuar el entrenamiento.
    """
    try:
        # Guardar estado del modelo y optimizador
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'model_config': {
                'text_hidden_size': 768,
                'image_hidden_size': 512,
                'combined_hidden_size': 256
            }
        }
        
        # Asegurar que el path sea relativo a MODEL_OUTPUT_FOLDER
        if not path.startswith(MODEL_OUTPUT_FOLDER):
            path = os.path.join(MODEL_OUTPUT_FOLDER, path)
            
        # Guardar checkpoint
        torch.save(checkpoint, path)
        
        # Guardar metadatos
        metadata = {
            'epoch': epoch,
            'loss': float(loss),  # Convert to float for JSON serialization
            'timestamp': str(datetime.datetime.now()),
            'device': str(device),
            'model_path': path
        }
        
        metadata_path = path.replace('.pth', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
            
        print(f"Checkpoint guardado en: {path}")
        print(f"Metadatos guardados en: {metadata_path}")
        return True
    except Exception as e:
        print(f"Error al guardar checkpoint: {str(e)}")
        return False

def load_checkpoint(path="multimodal_model.pth", model=None, optimizer=None):
    """
    Carga un checkpoint guardado previamente.
    """
    try:
        # Asegurar que el path sea relativo a MODEL_OUTPUT_FOLDER
        if not path.startswith(MODEL_OUTPUT_FOLDER):
            path = os.path.join(MODEL_OUTPUT_FOLDER, path)
            
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        metadata_path = path.replace('.pth', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                print(f"Checkpoint cargado de época {metadata['epoch']}")
                print(f"Loss: {metadata['loss']:.4f}")
                print(f"Guardado en: {metadata['timestamp']}")
        
        return checkpoint['epoch']
    except Exception as e:
        print(f"Error al cargar checkpoint: {str(e)}")
        return None

# Ejemplo de uso:
print("\nIniciando entrenamiento...")
train_model(
    model, 
    train_loader, 
    epochs=5, 
    save_path=os.path.join(MODEL_OUTPUT_FOLDER, "multimodal_model.pth"),
    save_interval=1
)

print(f"\nModelos guardados en: {MODEL_OUTPUT_FOLDER}")
print("Archivos disponibles:")
for f in os.listdir(MODEL_OUTPUT_FOLDER):
    print(f"- {f}")

# Para cargar un checkpoint:
# last_epoch = load_checkpoint("multimodal_model_final.pth", model, optimizer)
