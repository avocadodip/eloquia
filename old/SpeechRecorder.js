import React, { useState, useEffect } from 'react';
import { View, Button, Text } from 'react-native';
import { Audio } from 'expo-av';

const SpeechRecorder = ({ onRecordingComplete }) => {
  const [recording, setRecording] = useState();
  const [isRecording, setIsRecording] = useState(false);

  useEffect(() => {
    (async () => {
      await Audio.requestPermissionsAsync();
    })();
  }, []);

  const startRecording = async () => {
    try {
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      }); 
      const { recording } = await Audio.Recording.createAsync(
         Audio.RECORDING_OPTIONS_PRESET_HIGH_QUALITY
      );
      setRecording(recording);
      setIsRecording(true);
    } catch (err) {
      console.error('Failed to start recording', err);
    }
  };

  const stopRecording = async () => {
    recording.stopAndUnloadAsync();
    const uri = recording.getURI(); 
    setRecording(undefined);
    setIsRecording(false);
    onRecordingComplete(uri);
  };

  return (
    <View>
      <Button
        title={isRecording ? 'Stop Recording' : 'Start Recording'}
        onPress={isRecording ? stopRecording : startRecording}
      />
      {isRecording && <Text>Recording...</Text>}
    </View>
  );
};

export default SpeechRecorder;
