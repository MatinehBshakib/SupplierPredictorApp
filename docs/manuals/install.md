### Update pip to latest version
install python3.10<br/>
python3.10 -m pip install --upgrade pip

# Install required packages with specific versions
python3.10 -m pip install numpy==1.24.2<br/>
python3.10 -m pip install pandas==1.5.3<br/>
python3.10 -m pip install Django==4.2<br/>
python3.10 -m pip install requests==2.31.0<br/>
python3.10 -m pip install spacy==3.7.1<br/>
python3.10 -m pip install geopy==2.3.0<br/>
pip install nltk<br/>
python -m nltk.downloader stopwords<br/>
pip install scikit-learn<br/>
pip install sentence-transformers==2.2.2<br/>
pip install imbalanced-learn<br/>
pip install matplotlib<br/>
pip install xgboost<br/>
pip install joblib<br/>



# Download spaCy language model
python3.10 -m spacy download en_core_web_sm

### Verify installations
python3.10 -m pip list