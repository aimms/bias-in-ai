# Bias in AI

[![Downloads](https://img.shields.io/github/downloads/aimms/bias-in-ai/total?style=for-the-badge&logo=github&labelColor=000081&color=1847c9)](https://github.com/aimms/bias-in-ai/releases)
![AIMMS Version](https://img.shields.io/badge/AIMMS-24.5-white?style=for-the-badge&labelColor=009B00&color=00D400)
![WebUI Version](https://img.shields.io/badge/WebUI-24.11.2.3-white?style=for-the-badge&labelColor=009B00&color=00D400)
![AimmsDEX Version](https://img.shields.io/badge/AimmsDEX-24.4.1.3-white?style=for-the-badge&labelColor=009B00&color=00D400)

This repository contains a functional AIMMS example model for **Bias in AI**. It demonstrates how to connect AIMMS to a Python machine learning service and expose potential bias in AI-driven toxicity classification.

## 🎯 Business Problem

The combination of machine learning and everyday applications is at the heart of modern tech advancements. But hidden beneath its brilliance is a complex issue — bias within these algorithms.

This example illustrates bias by creating an AIMMS front-end to an existing Python application based on [Kaggle's AI Ethics course](https://www.kaggle.com/code/alexisbcook/identifying-bias-in-ai/tutorial). The application:

- Accepts a user-entered comment and evaluates its toxicity.
- Reads in a training dataset of comments with toxicity labels.
- Passes both the training data and the new comment to a Python service.
- Returns whether the comment is considered toxic — revealing that the underlying model may treat identical concepts differently depending on the demographic group referenced.

> **Note:** Bias can be observed in practice by entering words like `black` (marked **toxic**) versus `white` (marked **not toxic**).

This is also a relevant concern for Decision Support applications: basing decisions on data that is not representative of your market leads to poor outcomes.

## 📖 How to Use This Example

To get the most out of this model, we highly recommend reading our detailed step-by-step guide on the AIMMS How-To website:

👉 **[Read the Full Article: Bias in AI](https://how-to.aimms.com/Articles/623/623-bias-in-ai.html)**

### Prerequisites
- **AIMMS:** You will need AIMMS installed to run the model. An [AIMMS Community License](https://www.aimms.com/platform/aimms-community-edition/) is sufficient. [Download the Free Academic Edition here](https://www.aimms.com/support/licensing/) if you are a student.
- **Python 3.11:** Required to run the Python toxicity classification service. PyCharm is recommended but not required.
- **WebUI:** This model is optimized for the AIMMS WebUI for a modern, browser-based experience.

## 🚀 Getting Started

1. **Download the Release:** Go to the [Releases](https://github.com/aimms/bias-in-ai/releases) page and download the `.zip` file from the latest version.
2. **Open the Project:** Launch the `.aimms` file.
3. **Start the Python Service:** Follow the article instructions to start the Python backend locally or on AIMMS Cloud.
4. **Explore Bias:** Use the WebUI to import the dataset, enter comments, and observe the toxicity classification results.

## 🤝 Support & Feedback

This example is maintained by the **AIMMS User Support Team**.
- Found an issue? [Open an issue](https://github.com/aimms/bias-in-ai/issues).
- Questions? Reach out via the [AIMMS Community](https://community.aimms.com).

---
*Maintained by the AIMMS User Support Team. We optimize the way you build optimization.*
