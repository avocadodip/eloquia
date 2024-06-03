import React from 'react';
import { StyleSheet, View } from 'react-native';
import ClassifierScreen from './screens/ClassifierScreen';

export default function App() {
  return (
    <View style={styles.container}>
      <ClassifierScreen />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    justifyContent: 'center',
    alignItems: 'center',
  },
});
