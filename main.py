from mod.classifier import predict_paper_publishability, predict_conference_reasoning

# Task1: Classifying Research Papers as Publishable or Non-Publishable Using Feature Extraction and Clustering

path = r"sample.pdf"
sample_groq_api_key = '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'

publishability = predict_paper_publishability(path)

if publishability:
    conference, rationale = predict_conference_reasoning(path, sample_groq_api_key)
    print(f"The research paper is publishable and should be submitted to the {conference} conference.")
    print(f"Rationale: {rationale}")
else:
    print("The research paper is not publishable.")
