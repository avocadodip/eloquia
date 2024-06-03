import React, { useState } from 'react';
import { StyleSheet, Text, View, TouchableOpacity, Alert } from 'react-native';
import TorchModule from './TorchModule';

const App = () => {
  const [classificationResult, setClassificationResult] = useState('');

  const classifySpeech = async () => {
    try {
      const result = await TorchModule.classifySpeech();
      setClassificationResult(result);
    } catch (error) {
      Alert.alert("Error", "Failed to classify speech.");
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>Speech Disfluency Classifier</Text>
      <TouchableOpacity style={styles.button} onPress={classifySpeech}>
        <Text style={styles.buttonText}>Classify Speech</Text>
      </TouchableOpacity>
      <Text style={styles.result}>{classificationResult}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  title: {
    fontSize: 24,
    marginBottom: 20,
  },
  button: {
    backgroundColor: '#007bff',
    padding: 15,
    borderRadius: 5,
  },
  buttonText: {
    color: '#fff',
    fontSize: 18,
  },
  result: {
    marginTop: 20,
    fontSize: 20,
  },
});

export default App;
