import chromadb
from sentence_transformers import SentenceTransformer

class MedicalLogicBridge:
    def __init__(self, knowledge_file="medical_rules.txt"):
        print("🧠 Initializing Neuro-Symbolic Logic Bridge...")
        
        # 1. Setup Vector Database (ChromaDB - runs locally)
        self.chroma_client = chromadb.Client()
        try:
            self.collection = self.chroma_client.create_collection(name="leukemia_rules")
        except:
            self.collection = self.chroma_client.get_collection(name="leukemia_rules")

        # 2. Load Embedding Model (converts text to numbers)
        # 'all-MiniLM-L6-v2' is small, fast, and accurate for this task
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Ingest Knowledge
        self._ingest_knowledge(knowledge_file)

    def _ingest_knowledge(self, filepath):
        """Reads the text file and stores rules in the Vector DB."""
        if self.collection.count() > 0:
            print("   ✅ Knowledge Base already loaded.")
            return

        print(f"   📖 Reading rules from {filepath}...")
        with open(filepath, 'r') as f:
            rules = [line.strip() for line in f if line.strip().startswith("RULE:")]
        
        if not rules:
            print("   ⚠️ No rules found! Make sure lines start with 'RULE:'")
            return

        # Embed and Store
        embeddings = self.embedder.encode(rules).tolist()
        ids = [f"rule_{i}" for i in range(len(rules))]
        
        self.collection.add(
            documents=rules,
            embeddings=embeddings,
            ids=ids
        )
        print(f"   ✅ Successfully indexed {len(rules)} medical rules.")

    def get_supporting_evidence(self, prediction_class):
        # OLD: query = prediction_class 
        # NEW: Ask specifically about the defining features
        query = f"morphological features of {prediction_class} size chromatin nucleoli"
        
        results = self.collection.query(
            query_embeddings=self.embedder.encode([query]).tolist(),
            n_results=1 # You could also increase this to 3 to get more rules
        )
        return results['documents'][0][0]

    def verify_prediction(self, prediction_class, visual_features):
        """
        The Symbolic Check: Does the image match the rule?
        (In a full system, 'visual_features' would come from the CNN. 
         Here we simulate the logic check).
        """
        rule = self.get_supporting_evidence(prediction_class)
        
        print(f"\n🔎 SYMBOLIC VERIFICATION:")
        print(f"   🤖 Neural Prediction: {prediction_class}")
        print(f"   📜 Retrieved Rule:    {rule}")
        
        # Robust Logic Matching
        # We normalize both the rule and features to lowercase and check for key phrases
        rule_lower = rule.lower()
        matches = []
        
        # Stopwords: Generic terms that shouldn't trigger a match on their own
        # We want to match Adjectives (large, small, irregular), not Nouns (size, shape, cell)
        STOP_WORDS = {
            "size", "shape", "cell", "cells", "rule", "nuclear", "contours", 
            "chromatin", "nucleoli", "feature", "features", "of", "and", "the", "are"
        }
        
        for feature in visual_features:
            feature_lower = feature.lower()
            
            # 1. Exact Full Phrase Match (Best Case)
            if feature_lower in rule_lower:
                matches.append(feature)
                continue
                
            # 2. Smart Partial Match
            # Check individual words, BUT ignore generic stop words
            feature_words = feature_lower.split()
            valid_word_found = False
            
            for word in feature_words:
                if word in STOP_WORDS or len(word) <= 2:
                    continue
                
                if word in rule_lower:
                    valid_word_found = True
                    break # Found a significant adjective match (e.g. "small")
            
            if valid_word_found:
                 matches.append(feature)

        if len(matches) > 0:
            return True, f"Validated! Found matching evidence: {matches}"
        else:
            return False, f"⚠️ LO-SHOT INTERVENTION: Visual features {visual_features} do not match rule regarding '{prediction_class}'."

# --- TEST RUN ---
if __name__ == "__main__":
    # 1. Initialize
    bridge = MedicalLogicBridge()
    
    # 2. Simulate a Prediction from your Neural Network
    # Suppose the CNN predicts "Pro-B" and detects "large size"
    test_class = "Pro-B"
    test_features = ["large size", "prominent nucleoli"] 
    
    # 3. Run Verification
    is_valid, explanation = bridge.verify_prediction(test_class, test_features)
    print(f"   ✅ Verdict: {explanation}")