# Task Description

**With Sense** is a fictional photography hosting service created for educational purposes. Users upload photographs along with detailed descriptions (including shooting location, camera model, etc.), and additional descriptions can be contributed by other users.

For example, a photograph might have a description such as:  
`A hiker poses for a picture in front of stunning mountains and clouds.`

The goal of this task is to develop a **Proof of Concept (PoC)** for a reference image search tool that allows users to enter a textual query (e.g., `A man is crossing a mountain pass on a metal bridge.`) and retrieves images that best match the description by computing a similarity score between the image and text representations.

### Key Aspects of the Task

- **Image and Text Vectorization:**  
  - Text descriptions will be vectorized using techniques such as TF-IDF, BERT, or Word2Vec.
  - Images will be vectorized using a model like ResNet50 (implemented via PyTorch or Keras).
  - The similarity between text and image vectors will be measured with a score ranging from **0 to 1**.

- **Legal Restrictions:**  
  - In regions where local laws restrict processing content involving children (defined as individuals under 16) without explicit consent, the system must display a disclaimer instead of showing the image:
    > `This image is unavailable in your country in compliance with local laws.`  
  - For this PoC, any query containing prohibited content should trigger this disclaimer rather than returning an image.
  - As part of data preparation, any image-description pairs that include legally restricted content must be filtered out.

- **Data Description:**  
  - **Training Data:**  
    - `train_dataset.csv` includes image filenames, description identifiers (formatted as `<image_filename>#<description_number>`), and text descriptions (with up to five descriptions per image).  
    - `train_images/` contains the corresponding training images.
    - `CrowdAnnotations.tsv` and `ExpertAnnotations.tsv` provide crowdsourced and expert-rated data on the correspondence between images and descriptions.
  - **Testing Data:**  
    - `test_queries.csv` contains query IDs, textual queries, and the relevant image identifier.
    - `test_images/` includes images for evaluating model performance.

- **Model Development:**  
  - The PoC will use the best-performing model to generate vector representations for both images and text.
  - The model will concatenate these vectors and predict a similarity score or expert rating that indicates how well the text matches the image.
  - Various approaches (e.g., Linear Regression and Fully Connected Neural Networks) will be explored to determine the most effective method.

- **Demonstration:**  
  - A demonstration version will be created to showcase the feasibility of the image search by textual query. This demo will involve:
    - Testing the trained model on a set of predefined queries.
    - Implementing a function that takes a textual description, vectorizes it, and retrieves the most relevant image (or returns the legal disclaimer if restricted content is detected).

This task represents an experimental step toward building a robust text-based image search service that adheres to legal guidelines while providing accurate and relevant search results.
