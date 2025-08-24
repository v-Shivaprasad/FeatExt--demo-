import langextract as lx
import textwrap
from dotenv import load_dotenv
import os

api_key = os.getenv("GEMINI_API")
prompt = textwrap.dedent("""\
Extract homeopathic medicine information including remedies, symptoms, dosages, 
modalities (what makes symptoms better/worse), causations, and constitutional features.
Use exact text for extractions. Do not paraphrase or overlap entities.
Provide meaningful attributes for each entity to add context.""")

extraction_classes = [
    "medicine",
    "symptom", 
    "modality",
    "dosage",
    "causation",
    "constitutional_feature",
    "condition",
    "medicine_relationship"
]

examples = [
    lx.data.ExampleData(
        text=(
            "Arsenicum Album 30C, given twice daily, is excellent for food poisoning "
            "with burning pains in stomach, worse after midnight, better from warm drinks. "
            "The patient is restless, anxious, and chilly."
        ),
        extractions=[
            lx.data.Extraction(
                extraction_class="medicine",
                extraction_text="Arsenicum Album",
                attributes={
                    "potency": "30C",
                    "source": "mineral",
                    "constitutional_type": "anxious, chilly"
                }
            ),
            lx.data.Extraction(
                extraction_class="dosage",
                extraction_text="30C, given twice daily",
                attributes={
                    "potency": "30C",
                    "frequency": "twice daily"
                }
            ),
            lx.data.Extraction(
                extraction_class="condition",
                extraction_text="food poisoning",
                attributes={
                    "system_affected": "digestive",
                    "stage": "acute"
                }
            ),
            lx.data.Extraction(
                extraction_class="symptom",
                extraction_text="burning pains in stomach",
                attributes={
                    "character": "burning",
                    "location": "stomach",
                    "body_system": "digestive"
                }
            ),
            lx.data.Extraction(
                extraction_class="modality",
                extraction_text="worse after midnight",
                attributes={
                    "effect": "worse",
                    "trigger_type": "time",
                    "specific_trigger": "after midnight"
                }
            ),
            lx.data.Extraction(
                extraction_class="modality",
                extraction_text="better from warm drinks",
                attributes={
                    "effect": "better",
                    "trigger_type": "food/drink",
                    "specific_trigger": "warm drinks"
                }
            )
        ]
    )
]


with open("temp.txt", "r", encoding="utf-8") as file:
    input_text = file.read()

# help(lx.extract)
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-pro",
    api_key=api_key
)

lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl")

