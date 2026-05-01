import base64
import os
from pathlib import Path
from PIL import Image
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def pil_to_base64(img: Image.Image) -> str:
    import io
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def is_ski_mask(img: Image.Image) -> bool:
    """
    Returns True if the person is wearing a ski mask / balaclava.
    """

    image_b64 = pil_to_base64(img)

    prompt = """
You are a strict visual classifier.

Task:
Determine whether the visible face is wearing a ski mask or balaclava.

Definition:
A ski mask/balaclava covers most of the face, leaving only the eyes
(or eyes and small mouth opening) visible.

Rules:
- Ignore hats, helmets, sunglasses.
- Medical masks are NOT ski masks.
- Only answer based on visible evidence.
- If unsure, return false.

Output format:
Return ONLY:

{"ski_mask": true}
or
{"ski_mask": false}
"""

    response = client.responses.create(
        model="gpt-4.1-mini",   # fast + cheap + strong vision
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{image_b64}",
                },
            ],
        }],
        temperature=0,
    )

    text = response.output_text.strip()

    import json
    result = json.loads(text)

    return bool(result["ski_mask"])