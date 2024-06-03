import React, { useState } from 'react';
import { View, Text, StyleSheet, Button } from 'react-native';
import SpeechRecorder from './SpeechRecorder';
import TorchModule from './TorchModule';

const ClassifierScreen = () => {
  const [classification, setClassification] = useState('');

  const handleRecordingComplete = async (uri) => {
    const result = await TorchModule.classifySpeech(uri);
    setClassification(result);
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Disfluency Speech Classifier</Text>
      <SpeechRecorder onRecordingComplete={handleRecordingComplete} />
      <Text style={styles.classificationText}>{classification}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    fontSize: 22,
    marginBottom: 20,
  },
  classificationText: {
    marginTop: 20,
    fontSize: 18,
  },
});

export default ClassifierScreen;
