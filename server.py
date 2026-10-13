import asyncio
import base64
import binascii
import json
import logging
import socket
import time
import threading
import os
import uuid
from pymongo import MongoClient

from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, List, Dict
from pathlib import Path

import cv2
import numpy as np
import face_recognition
from ultralytics import YOLO
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from zeroconf.asyncio import AsyncZeroconf
from zeroconf import ServiceInfo

 
# CONFIGURATION & LOGGING
 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sacsia.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("SACSIA-Server")

mongo_client = MongoClient("mongodb://localhost:27017/")
db = mongo_client["sacsia_db"]
members_collection = db["members"]

UPLOAD_DIR = "db/uploads/members"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Modèles YOLO
YOLO_MODEL_PATH = r"runs\facial_recognition\train-6\weights\best.pt"
YOLO_PARKING_MODEL_PATH = r"runs\detect\model_for_parking\weights\best.pt"

yolo_model: Optional[YOLO] = None

 
# MODÈLES DE DONNÉES
 
@dataclass
class Agent:
    id: str
    name: str
    role: str
    matricule: str
    department: str
    badge_uid: Optional[str] = None
    fingerprint_id: Optional[int] = None
    encoding: Optional[np.ndarray] = None
    photo_path: Optional[str] = None
    is_active: bool = True

@dataclass
class AccessLog:
    id: str
    timestamp: str
    agent_id: str
    agent_name: str
    agent_matricule: str
    method: str
    face_match: bool
    liveness_passed: bool
    spoof_detected: bool
    spoof_reason: str
    status: str
    duration_ms: float = 0.0

known_agents_db: Dict[str, Agent] = {}
access_history: List[AccessLog] = []

def load_known_agents():
    try:
        members = list(members_collection.find({"is_active": True}))
        for m in members:
            agent_id = m.get("matricule")
            encoding = np.array(m["face_encoding"]) if "face_encoding" in m and m["face_encoding"] else None
            
            known_agents_db[agent_id] = Agent(
                id=agent_id,
                name=m.get("full_name"),
                role=m.get("role"),
                matricule=m.get("matricule"),
                department=m.get("department"),
                fingerprint_id=m.get("fingerprint_id"),
                encoding=encoding,
                photo_path=m.get("photo_path"),
                is_active=m.get("is_active", True)
            )
        logger.info(f"✅ {len(known_agents_db)} membres chargés depuis MongoDB")
    except Exception as e:
        logger.error(f"❌ Erreur de chargement des membres depuis MongoDB : {e}")

def find_agent_by_id(user_key) -> Optional[Agent]:
    user_key = str(user_key)
    if user_key in known_agents_db:
        return known_agents_db[user_key]
    
    fp_id = None
    if user_key.startswith("USER_FP_"):
        try:
            fp_id = int(user_key.replace("USER_FP_", ""))
        except ValueError:
            pass
    elif user_key.isdigit():
        fp_id = int(user_key)

    if fp_id is not None:
        for agent in known_agents_db.values():
            if str(agent.fingerprint_id) == str(fp_id):
                return agent
    return None

def generate_test_logs():
    import random
    if not known_agents_db:
        logger.warning("⚠️ Aucun membre actif trouvé dans MongoDB.")
        return

    methods = ['fingerprint', 'badge', 'qr_code']
    statuses = ['GRANTED', 'DENIED', 'SPOOF_ALERT']
    status_weights = [0.85, 0.10, 0.05]
    now = datetime.now()

    for i in range(50):
        agent = random.choice(list(known_agents_db.values()))
        timestamp = now - __import__('datetime').timedelta(minutes=random.randint(1, 10080))
        status = random.choices(statuses, weights=status_weights)[0]

        log = AccessLog(
            id=f"LOG_{timestamp.strftime('%Y%m%d%H%M%S')}_{i:04d}",
            timestamp=timestamp.isoformat(),
            agent_id=agent.id,
            agent_name=agent.name,
            agent_matricule=agent.matricule,
            method=random.choice(methods),
            face_match=(status == 'GRANTED'),
            liveness_passed=(status != 'SPOOF_ALERT'),
            spoof_detected=(status == 'SPOOF_ALERT'),
            spoof_reason="Forme rectangulaire détectée" if status == 'SPOOF_ALERT' else "",
            status=status,
            duration_ms=random.uniform(800, 3500)
        )
        access_history.append(log)

    access_history.sort(key=lambda x: x.timestamp, reverse=True)

 
# VISION PAR ORDINATEUR & ANTI-SPOOFING
 
def check_passive_liveness(face_crop: np.ndarray) -> tuple[bool, str]:
    if face_crop.size == 0:
        return False, "Visage vide"
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 20.0:
        return False, f"Image trop floue (Laplacian: {laplacian_var:.1f})"
    contrast = float(gray.std())
    if contrast < 12.0:
        return False, f"Contraste insuffisant ({contrast:.1f})"
    return True, "Qualité image acceptable"

def detect_presentation_attack(frame: np.ndarray, face_box: tuple) -> tuple[bool, str]:
    x1, y1, x2, y2 = face_box
    h, w = frame.shape[:2]
    pad = 30
    x1_p = max(0, x1 - pad)
    y1_p = max(0, y1 - pad)
    x2_p = min(w, x2 + pad)
    y2_p = min(h, y2 + pad)
    roi = frame[y1_p:y2_p, x1_p:x2_p]
    if roi.size == 0:
        return False, ""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 20.0:
        return True, "Texture du visage insuffisante"
    _, bright = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)
    bright_ratio = cv2.countNonZero(bright) / float(roi.shape[0] * roi.shape[1])
    if bright_ratio > 0.20:
        return True, "Surface fortement réfléchissante détectée"
    return False, ""

def extract_face_encoding(image_path: str) -> Optional[List[float]]:
    try:
        image = cv2.imread(image_path)
        if image is None:
            return None
        if yolo_model is not None:
            results = yolo_model.predict(source=image, conf=0.6, verbose=False)
            boxes = results[0].boxes
            if len(boxes) > 0:
                box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(rgb_image, [(y1, x2, y2, x1)])
                if len(encodings) > 0:
                    return encodings[0].tolist()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_image)
        if len(encodings) > 0:
            return encodings[0].tolist()
        return None
    except Exception as e:
        logger.error(f"Erreur extraction faciale: {e}")
        return None

def decode_camera_image(base64_image: str) -> np.ndarray:
    if not isinstance(base64_image, str):
        raise ValueError("Image absente ou format invalide.")
    base64_image = base64_image.strip()
    if not base64_image:
        raise ValueError("Image Base64 vide.")
    if "," in base64_image and base64_image.startswith("data:"):
        base64_image = base64_image.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(base64_image, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("Image Base64 invalide.")
    if not image_bytes:
        raise ValueError("Aucune donnée image reçue.")
    np_arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Impossible de décoder l'image caméra.")
    return frame

def process_facial_recognition(base64_image: str, expected_user_id: str) -> dict:
    start_time = time.time()
    try:
        try:
            frame = decode_camera_image(base64_image)
        except ValueError as e:
            return {"success": False, "reason": str(e), "details": {"stage": "decode", "duration_ms": (time.time() - start_time) * 1000}}
        
        if frame is None:
            return {"success": False, "reason": "Erreur de décodage d'image", "details": {"stage": "decode", "duration_ms": (time.time() - start_time) * 1000}}

        h, w = frame.shape[:2]
        if yolo_model is None:
            return {"success": False, "reason": "Modèle YOLO non chargé", "details": {"stage": "yolo_init", "duration_ms": (time.time() - start_time) * 1000}}

        results = yolo_model.predict(source=frame, conf=0.6, verbose=False)
        boxes = results[0].boxes
        if len(boxes) == 0:
            return {"success": False, "reason": "Aucun visage détecté par le modèle YOLO", "details": {"stage": "face_detection", "duration_ms": (time.time() - start_time) * 1000}}

        box = max(boxes, key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        face_box = (x1, y1, x2, y2)
        face_crop = frame[y1:y2, x1:x2]

        is_spoof, spoof_reason = detect_presentation_attack(frame, face_box)
        if is_spoof:
            return {"success": False, "reason": f"Usurpation détectée: {spoof_reason}", "details": {"stage": "anti_spoof", "duration_ms": (time.time() - start_time) * 1000}}

        is_real, liveness_reason = check_passive_liveness(face_crop)
        if not is_real:
            return {"success": False, "reason": f"Preuve de vie échouée: {liveness_reason}", "details": {"stage": "liveness", "duration_ms": (time.time() - start_time) * 1000}}

        if expected_user_id not in known_agents_db:
            return {"success": False, "reason": "Agent inconnu dans la base", "details": {"stage": "identity", "duration_ms": (time.time() - start_time) * 1000}}

        agent = known_agents_db[expected_user_id]
        if agent.encoding is None:
            return {"success": False, "reason": "Abonné sans empreinte faciale enregistrée", "details": {"stage": "identity", "duration_ms": (time.time() - start_time) * 1000}}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        crop_encodings = face_recognition.face_encodings(rgb_frame, [(y1, x2, y2, x1)])
        if len(crop_encodings) == 0:
            return {"success": False, "reason": "Impossible de lire le descripteur du visage", "details": {"stage": "encoding", "duration_ms": (time.time() - start_time) * 1000}}

        distance = face_recognition.face_distance([agent.encoding], crop_encodings[0])[0]
        match = distance < 0.52

        if match:
            return {"success": True, "reason": "Identité confirmée", "details": {"stage": "complete", "distance": float(distance), "liveness": "passed", "duration_ms": (time.time() - start_time) * 1000}}
        else:
            return {"success": False, "reason": "Visage non correspondant", "details": {"stage": "matching", "distance": float(distance), "duration_ms": (time.time() - start_time) * 1000}}
    except Exception as e:
        return {"success": False, "reason": f"Erreur serveur: {str(e)}", "details": {"stage": "error", "duration_ms": (time.time() - start_time) * 1000}}

 
# GESTIONNAIRE DE CONNEXION
 
esp32_command_queue = asyncio.Queue()

class ConnectionManager:
    def __init__(self):
        self.mobile_ws: Optional[WebSocket] = None
        self.web_clients: List[WebSocket] = []

    async def connect_mobile(self, websocket: WebSocket):
        await websocket.accept()
        self.mobile_ws = websocket
        await self.broadcast_to_web({"event": "MOBILE_STATUS", "status": "online"})

    async def disconnect_mobile(self):
        self.mobile_ws = None
        await self.broadcast_to_web({"event": "MOBILE_STATUS", "status": "offline"})

    async def send_to_mobile(self, message: dict):
        if self.mobile_ws:
            try:
                await self.mobile_ws.send_json(message)
            except Exception:
                pass

    async def connect_web(self, websocket: WebSocket):
        await websocket.accept()
        self.web_clients.append(websocket)
        await websocket.send_json({"event": "MOBILE_STATUS", "status": "online" if self.mobile_ws is not None else "offline"})
        await websocket.send_json({"event": "HISTORY_INIT", "logs": [asdict(log) for log in access_history[:50]], "stats": self.get_stats()})

    def disconnect_web(self, websocket: WebSocket):
        if websocket in self.web_clients:
            self.web_clients.remove(websocket)

    async def broadcast_to_web(self, message: dict):
        dead = []
        for ws in self.web_clients:
            try:
                await ws.send_json(message)
            except:
                dead.append(ws)
        for ws in dead:
            if ws in self.web_clients:
                self.web_clients.remove(ws)

    def get_stats(self) -> dict:
        granted = sum(1 for l in access_history if l.status == 'GRANTED')
        denied = sum(1 for l in access_history if l.status == 'DENIED')
        spoof = sum(1 for l in access_history if l.status == 'SPOOF_ALERT')
        return {"total": len(access_history), "granted": granted, "denied": denied, "spoof": spoof}

manager = ConnectionManager()

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

local_ip = get_local_ip()
aiozeroconf: Optional[AsyncZeroconf] = None
service_info: Optional[ServiceInfo] = None

def start_udp_discovery():
    def udp_server():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            sock.bind(('0.0.0.0', 8001))
        except OSError:
            return
        sock.settimeout(2.0)
        while True:
            try:
                data, addr = sock.recvfrom(1024)
                if data.decode().strip() == 'SACSIA_DISCOVER':
                    sock.sendto(f"SACSIA_SERVER|{local_ip}|8000".encode(), addr)
            except:
                continue
    threading.Thread(target=udp_server, daemon=True).start()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global aiozeroconf, yolo_model, local_ip, service_info
    local_ip = get_local_ip()
    try:
        yolo_model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        logger.error(f"Erreur chargement YOLO: {e}")

    load_known_agents()
    generate_test_logs()
    start_udp_discovery()

    try:
        service_info = ServiceInfo("_sacsia._tcp.local.", "SACSIA-Server._sacsia._tcp.local.", addresses=[socket.inet_aton(local_ip)], port=8000, properties={"version": "1.0"}, server="sacsia-server.local.")
        aiozeroconf = AsyncZeroconf()
        await aiozeroconf.async_register_service(service_info)
    except Exception:
        pass

    yield

    if aiozeroconf and service_info:
        try:
            await aiozeroconf.async_unregister_service(service_info)
        except Exception:
            pass
        await aiozeroconf.async_close()

app = FastAPI(title="SACSIA Access Control Server", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/db/uploads", StaticFiles(directory="db/uploads"), name="uploads")

 
# FLUX VIDÉO PARKING (ORIGINAL & DIRECT)
 
def generate_parking_frames():
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    yolo_parking_model = YOLO(YOLO_PARKING_MODEL_PATH)

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(1)
            continue
            
        results = yolo_parking_model.predict(frame, conf=0.4, verbose=False)
        annotated_frame = results[0].plot()

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.get("/api/parking/stream")
async def parking_video_stream():
    return StreamingResponse(generate_parking_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

 
# ENDPOINTS HTTP & WEBSOCKETS STANDARDS
 
@app.get("/api/discover")
async def discover():
    return {"service": "SACSIA", "version": "1.0", "server": "sacsia-server.local", "ip": local_ip}

@app.get("/api/agents")
async def list_agents():
    return {"agents": [{"id": a.id, "name": a.name, "role": a.role, "matricule": a.matricule, "department": a.department, "photo": a.photo_path, "is_active": a.is_active} for a in known_agents_db.values()]}

@app.get("/api/history")
async def get_history(limit: int = 50, offset: int = 0):
    total = len(access_history)
    logs = access_history[offset:offset + limit]
    return {"total": total, "offset": offset, "limit": limit, "logs": [asdict(log) for log in logs]}

@app.get("/api/stats")
async def get_stats():
    return manager.get_stats()

@app.post("/api/esp32/heartbeat")
async def esp32_heartbeat(data: dict):
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/api/esp32/poll")
async def esp32_poll():
    try:
        command = await asyncio.wait_for(esp32_command_queue.get(), timeout=0.8)
        return command
    except asyncio.TimeoutError:
        return {"action": "NONE"}

@app.post("/api/esp32/scan")
async def esp32_scan(data: dict):
    user_id = data.get("user_id", "UNKNOWN")
    agent = find_agent_by_id(user_id)
    if not agent:
        await manager.send_to_mobile({"event": "AUTH_RESULT", "status": "failed", "success": False, "message": "❌ Accès Refusé\nEmpreinte non enregistrée."})
        await esp32_command_queue.put({"action": "DENY", "reason": "Agent Inconnu"})
        return {"status": "error", "message": "Agent Inconnu"}

    await manager.send_to_mobile({
        "event": "STEP_CHANGE", "step": "face_recognition", "auth_method": "FINGERPRINT",
        "message": f"Empreinte de {agent.name} validée.\nPositionnez votre visage.", "user_id": agent.matricule, "agent_name": agent.name
    })
    return {"status": "ok", "next_step": "FACE_REQUIRED", "agent_name": agent.name}

@app.websocket("/ws/mobile")
async def websocket_mobile(websocket: WebSocket):
    await manager.connect_mobile(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if not data:
                continue
            payload = json.loads(data)
            event = payload.get("event") or payload.get("action")
            
            if event == "VERIFY_QR":
                matricule = payload.get("matricule")
                agent = find_agent_by_id(matricule)
                if not agent:
                    await manager.send_to_mobile({"event": "AUTH_RESULT", "status": "failed", "success": False, "message": "❌ Agent non reconnu."})
                    continue
                await manager.send_to_mobile({
                    "event": "STEP_CHANGE", "step": "face_recognition", "auth_method": "QR_CODE",
                    "message": f"Badge validé. Alignez votre visage.", "user_id": agent.matricule, "agent_name": agent.name
                })
            elif event == "PROCESS_FACE":
                base64_img = payload.get("image")
                raw_user_id = payload.get("user_id", "UNKNOWN")
                start_time = time.time()
                agent = find_agent_by_id(raw_user_id)
                if not agent:
                    continue
                result = process_facial_recognition(base64_img, agent.matricule)
                duration = (time.time() - start_time) * 1000
                
                log_entry = AccessLog(
                    id=f"LOG_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    timestamp=datetime.now().isoformat(), agent_id=agent.id, agent_name=agent.name,
                    agent_matricule=agent.matricule, method="fingerprint", face_match=result["success"],
                    liveness_passed=result["details"].get("liveness") == "passed",
                    spoof_detected=not result["success"] and "spoof" in result.get("reason", "").lower(),
                    spoof_reason=result["details"].get("spoof_type", ""),
                    status="GRANTED" if result["success"] else ("SPOOF_ALERT" if "spoof" in result.get("reason", "").lower() else "DENIED"),
                    duration_ms=duration
                )
                access_history.insert(0, log_entry)
                
                if result["success"]:
                    await manager.send_to_mobile({"event": "AUTH_RESULT", "status": "success", "success": True, "message": f"✅ Accès Autorisé !\n{agent.name}"})
                    await esp32_command_queue.put({"action": "UNLOCK", "duration": 5})
                else:
                    await manager.send_to_mobile({"event": "AUTH_RESULT", "status": "failed", "success": False, "message": f"❌ Refusé\n{result['reason']}"})
                    await esp32_command_queue.put({"action": "DENY"})
                await manager.broadcast_to_web({"event": "NEW_ACCESS", "log": asdict(log_entry), "stats": manager.get_stats()})
    except WebSocketDisconnect:
        await manager.disconnect_mobile()

@app.websocket("/ws/web")
async def websocket_web(websocket: WebSocket):
    await manager.connect_web(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if not data:
                continue
            payload = json.loads(data)
            event = payload.get("event") or payload.get("action")
            if event == "GET_STATS":
                await websocket.send_json({"event": "STATS_UPDATE", "stats": manager.get_stats()})
    except WebSocketDisconnect:
        manager.disconnect_web(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)