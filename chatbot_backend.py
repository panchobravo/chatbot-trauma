# Mapeo de Jerga -> Español Neutro
CHILENISMOS_MAP = {
    # Typos y Expresiones Graves
    r"\bquiero la pata\b": "quebre la pierna", 
    r"\bme quiero\b": "me quebre", 
    r"\bme saque la cresta\b": "caida grave",
    r"\bme saque la chucha\b": "caida grave",
    
    # Anatomía y Objetos
    r"\bpata\b": "pierna", r"\bguata\b": "estomago", r"\bpucho\b": "cigarro",
    r"\bfumo\b": "tabaco", r"\bfumas\b": "tabaco", r"\bfumar\b": "tabaco",
    
    # Insultos = Dolor/Frustración (PARA QUE NO DE FALLBACK)
    r"\bputa la wea\b": "estoy mal",
    r"\bconchesumadre\b": "dolor terrible",
    r"\bconchetumare\b": "dolor terrible",
    r"\bctm\b": "dolor terrible",
    r"\b(c+s+m+)\b": "dolor terrible", # Atrapa "csmmmmm"
    r"\bmierda\b": "dolor",
    r"\bwea\b": "cosa",
    
    # Intensidad
    r"\bcago\b": "daño", r"\bcaleta\b": "mucho", r"\bbrigido\b": "intenso", 
    r"\bpal gato\b": "mal", r"\bmas o menos\b": "regular", r"\bmaoma\b": "regular",
    r"\bhecho bolsa\b": "muy mal", r"\bcuatico\b": "grave",
    
    # Modismos Positivos
    r"\bjoya\b": "excelente", r"\bfilete\b": "excelente", r"\bbacan\b": "excelente",
    r"\bseco\b": "experto",
    
    # Verbos
    r"\bcachai\b": "entiendes", r"\bpesca\b": "atencion", r"\bal tiro\b": "inmediatamente", 
    r"\bsipo\b": "si", r"\byapo\b": "ya", r"\bnopo\b": "no"
}

# Lista ampliada para detectar tono urgente
GROSERIAS_DOLOR = ["ctm", "conchetumare", "conchesumadre", "mierda", "pico", "puta", "recontra", "chucha", "csm"]
