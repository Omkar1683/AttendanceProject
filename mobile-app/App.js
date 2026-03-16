import React, { useState, useRef, useEffect } from 'react';
import {
  StyleSheet,
  Text,
  View,
  TextInput,
  TouchableOpacity,
  ScrollView,
  SafeAreaView,
  StatusBar,
  Image,
  Alert,
  ActivityIndicator,
  Modal
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { CameraView, useCameraPermissions } from 'expo-camera';
import {
  Camera,
  BarChart3,
  Home,
  Settings,
  ChevronLeft,
  Bell,
  Calendar,
  LogOut,
  ChevronDown,
  Download
} from 'lucide-react-native';
import { api } from './utils/api';

// --- MAIN APP COMPONENT ---
export default function App() {
  const [currentScreen, setCurrentScreen] = useState('Login');
  const [userRole, setUserRole] = useState('Teacher');
  const [userInfo, setUserInfo] = useState(null);
  const [selectedClass, setSelectedClass] = useState(null);
  const [currentSession, setCurrentSession] = useState(null);

  const navigateTo = (screen) => {
    setCurrentScreen(screen);
  };

  const handleLogin = (user) => {
    setUserInfo(user);
    if (user.role === 'teacher') {
      navigateTo('TeacherDashboard');
    } else {
      navigateTo('StudentDashboard');
    }
  };

  const handleLogout = async () => {
    await api.logout();
    setUserInfo(null);
    setSelectedClass(null);
    setCurrentSession(null);
    navigateTo('Login');
  };

  const renderScreen = () => {
    switch (currentScreen) {
      case 'Login':
        return <LoginScreen navigateTo={navigateTo} userRole={userRole} setUserRole={setUserRole} onLogin={handleLogin} />;
      case 'Signup':
        return <SignupScreen navigateTo={navigateTo} />;
      case 'TeacherDashboard':
        return <TeacherDashboard navigateTo={navigateTo} userInfo={userInfo} onLogout={handleLogout} setSelectedClass={setSelectedClass} setCurrentSession={setCurrentSession} />;
      case 'StudentDashboard':
        return <StudentDashboard navigateTo={navigateTo} userInfo={userInfo} onLogout={handleLogout} />;
      case 'StudentTimeline':
        return <StudentTimelineScreen navigateTo={navigateTo} />;
      case 'StudentSessions':
        return <StudentSessionsScreen navigateTo={navigateTo} />;
      case 'StudentProfile':
        return <StudentProfileScreen navigateTo={navigateTo} userInfo={userInfo} setUserInfo={setUserInfo} />;
      case 'ChangePassword':
        return <ChangePasswordScreen navigateTo={navigateTo} />;
      case 'RegisterStudent':
        return <RegisterStudentScreen navigateTo={navigateTo} />;
      case 'ScanAttendance':
        return <ScanAttendanceScreen navigateTo={navigateTo} currentSession={currentSession} />;
      case 'DetailedReport':
        return <DetailedReportScreen navigateTo={navigateTo} selectedClass={selectedClass} />;
      case 'NotificationHub':
        return <NotificationHubScreen navigateTo={navigateTo} selectedClass={selectedClass} />;
      default:
        return <LoginScreen navigateTo={navigateTo} userRole={userRole} setUserRole={setUserRole} onLogin={handleLogin} />;
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" backgroundColor="#f3f4f6" />
      <View style={styles.contentContainer}>
        {renderScreen()}
      </View>
    </SafeAreaView>
  );
}

// --- 1. LOGIN SCREEN ---
const LoginScreen = ({ navigateTo, userRole, setUserRole, onLogin }) => {
  const [email, setEmail] = useState('teacher@ves.ac.in');
  const [password, setPassword] = useState('vaishali123');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!email || !password) {
      Alert.alert('Error', 'Please enter email and password');
      return;
    }

    setLoading(true);
    try {
      const result = await api.login(email, password);
      onLogin(result.user);
    } catch (error) {
      Alert.alert('Login Failed', error.message || 'Invalid credentials');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.screenContainer}>
      <View style={styles.loginContent}>
        <Text style={styles.appTitle}>AttendAI</Text>
        <Text style={styles.appSubtitle}>Automatic Attendance System</Text>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>User ID / Email</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g., omkar@institute.edu"
            placeholderTextColor="#9ca3af"
            value={email}
            onChangeText={setEmail}
            autoCapitalize="none"
            keyboardType="email-address"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Password</Text>
          <TextInput
            style={styles.input}
            placeholder="••••••••"
            placeholderTextColor="#9ca3af"
            secureTextEntry
            value={password}
            onChangeText={setPassword}
          />
        </View>

        <View style={styles.roleContainer}>
          <Text style={styles.helperText}>Login as:</Text>
          <View style={styles.roleToggle}>
            <TouchableOpacity
              style={[styles.roleButton, userRole === 'Teacher' && styles.roleButtonActive]}
              onPress={() => setUserRole('Teacher')}
            >
              <Text style={[styles.roleText, userRole === 'Teacher' && styles.roleTextActive]}>Teacher</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.roleButton, userRole === 'Student' && styles.roleButtonActive]}
              onPress={() => setUserRole('Student')}
            >
              <Text style={[styles.roleText, userRole === 'Student' && styles.roleTextActive]}>Student</Text>
            </TouchableOpacity>
          </View>
        </View>

        <TouchableOpacity
          style={styles.primaryButton}
          onPress={handleLogin}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="white" />
          ) : (
            <Text style={styles.primaryButtonText}>Secure Login</Text>
          )}
        </TouchableOpacity>

        <TouchableOpacity
          style={{ marginTop: 16, alignItems: 'center' }}
          onPress={() => navigateTo('Signup')}
        >
          <Text style={{ color: '#2563eb', fontSize: 14 }}>Don't have an account? Sign Up</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

// --- 2. SIGNUP SCREEN ---
const SignupScreen = ({ navigateTo }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [role, setRole] = useState('teacher');
  const [department, setDepartment] = useState('');
  const [rollNo, setRollNo] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSignup = async () => {
    if (!email || !password || !name || !department) {
      Alert.alert('Error', 'Please fill all required fields');
      return;
    }

    if (role === 'student' && !rollNo) {
      Alert.alert('Error', 'Roll number is required for students');
      return;
    }

    setLoading(true);
    try {
      const result = await api.signup(
        email,
        password,
        name,
        role,
        department,
        role === 'student' ? rollNo : null
      );
      Alert.alert('Success', 'Account created successfully!', [
        { text: 'OK', onPress: () => navigateTo('Login') }
      ]);
    } catch (error) {
      Alert.alert('Signup Failed', error.message || 'Could not create account');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.screenContainer}>
      <ScrollView style={styles.scrollContent} contentContainerStyle={{ paddingBottom: 40 }}>
        <TouchableOpacity
          onPress={() => navigateTo('Login')}
          style={{ marginBottom: 20 }}
        >
          <Text style={{ color: '#2563eb', fontSize: 14 }}>← Back to Login</Text>
        </TouchableOpacity>

        <Text style={styles.appTitle}>Create Account</Text>
        <Text style={styles.appSubtitle}>Sign up for AttendAI</Text>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Full Name *</Text>
          <TextInput
            style={styles.input}
            placeholder="Enter your full name"
            placeholderTextColor="#9ca3af"
            value={name}
            onChangeText={setName}
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Email *</Text>
          <TextInput
            style={styles.input}
            placeholder="your.email@institute.edu"
            placeholderTextColor="#9ca3af"
            value={email}
            onChangeText={setEmail}
            autoCapitalize="none"
            keyboardType="email-address"
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Password *</Text>
          <TextInput
            style={styles.input}
            placeholder="••••••••"
            placeholderTextColor="#9ca3af"
            secureTextEntry
            value={password}
            onChangeText={setPassword}
          />
        </View>

        <View style={styles.inputGroup}>
          <Text style={styles.label}>Department *</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g., MCA, Computer Science"
            placeholderTextColor="#9ca3af"
            value={department}
            onChangeText={setDepartment}
          />
        </View>

        <View style={styles.roleContainer}>
          <Text style={styles.helperText}>Register as:</Text>
          <View style={styles.roleToggle}>
            <TouchableOpacity
              style={[styles.roleButton, role === 'teacher' && styles.roleButtonActive]}
              onPress={() => setRole('teacher')}
            >
              <Text style={[styles.roleText, role === 'teacher' && styles.roleTextActive]}>Teacher</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.roleButton, role === 'student' && styles.roleButtonActive]}
              onPress={() => setRole('student')}
            >
              <Text style={[styles.roleText, role === 'student' && styles.roleTextActive]}>Student</Text>
            </TouchableOpacity>
          </View>
        </View>

        {role === 'student' && (
          <View style={styles.inputGroup}>
            <Text style={styles.label}>Roll Number *</Text>
            <TextInput
              style={styles.input}
              placeholder="e.g., MCA001"
              placeholderTextColor="#9ca3af"
              value={rollNo}
              onChangeText={setRollNo}
            />
          </View>
        )}

        <TouchableOpacity
          style={styles.primaryButton}
          onPress={handleSignup}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="white" />
          ) : (
            <Text style={styles.primaryButtonText}>Create Account</Text>
          )}
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
};

// --- 3. TEACHER DASHBOARD ---
const TeacherDashboard = ({ navigateTo, userInfo, onLogout, setSelectedClass, setCurrentSession }) => {
  const [classes, setClasses] = useState([]);
  const [selectedClassId, setSelectedClassId] = useState(null);
  const [todaySummary, setTodaySummary] = useState(null);
  const [defaulters, setDefaulters] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showClassPicker, setShowClassPicker] = useState(false);

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    if (selectedClassId) {
      loadClassData(selectedClassId);
    }
  }, [selectedClassId]);

  const loadData = async () => {
    try {
      const classesResult = await api.getClasses(userInfo.id);
      if (classesResult.status === 'success' && classesResult.data.length > 0) {
        setClasses(classesResult.data);
        setSelectedClassId(classesResult.data[0].id);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to load classes');
    } finally {
      setLoading(false);
    }
  };

  const loadClassData = async (classId) => {
    try {
      const [summaryResult, defaultersResult] = await Promise.all([
        api.getTodayAnalytics(classId),
        api.getDefaulters(classId)
      ]);

      if (summaryResult.status === 'success') {
        setTodaySummary(summaryResult.data);
      }
      if (defaultersResult.status === 'success') {
        setDefaulters(defaultersResult.data);
      }
    } catch (error) {
      console.error('Error loading class data:', error);
    }
  };

  const handleStartAttendance = async () => {
    if (!selectedClassId) {
      Alert.alert('Error', 'Please select a class first');
      return;
    }

    try {
      const result = await api.createSession(selectedClassId, 'Room 504');
      if (result.status === 'success') {
        setCurrentSession(result.session_id);
        const selectedClassData = classes.find(c => c.id === selectedClassId);
        setSelectedClass(selectedClassData);
        navigateTo('ScanAttendance');
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to start session');
    }
  };

  const handleClassSelect = (classId) => {
    setSelectedClassId(classId);
    setShowClassPicker(false);
  };

  if (loading) {
    return (
      <View style={[styles.screenContainer, { justifyContent: 'center', alignItems: 'center' }]}>
        <ActivityIndicator size="large" color="#2563eb" />
      </View>
    );
  }

  const selectedClassData = classes.find(c => c.id === selectedClassId);

  return (
    <View style={styles.screenContainer}>
      {/* Class Picker Modal */}
      <Modal
        visible={showClassPicker}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setShowClassPicker(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Select Class</Text>
            <ScrollView style={{ maxHeight: 400, minHeight: 100 }}>
              {classes && classes.length > 0 ? (
                classes.map((cls) => (
                  <TouchableOpacity
                    key={cls.id}
                    style={[
                      styles.modalItem,
                      cls.id === selectedClassId && styles.modalItemSelected
                    ]}
                    onPress={() => handleClassSelect(cls.id)}
                  >
                    <Text style={styles.modalItemText}>{cls.name}</Text>
                    <Text style={styles.modalItemSubtext}>{cls.batch} • {cls.total_students} students</Text>
                  </TouchableOpacity>
                ))
              ) : (
                <View style={{ padding: 32, alignItems: 'center' }}>
                  <Text style={{ color: '#6b7280', fontSize: 14, textAlign: 'center' }}>No classes found.{'\n'}Please add classes first.</Text>
                </View>
              )}
            </ScrollView>
            <TouchableOpacity
              style={styles.modalCloseButton}
              onPress={() => setShowClassPicker(false)}
            >
              <Text style={styles.modalCloseButtonText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>Dashboard</Text>
          <Text style={styles.headerSubtitle}>Welcome, {userInfo?.name}</Text>
        </View>
        <TouchableOpacity onPress={onLogout}>
          <LogOut color="#4b5563" size={24} />
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scrollContent}>

        {/* Session Selector */}
        <View style={styles.card}>
          <Text style={styles.cardLabel}>SELECT SESSION</Text>
          <TouchableOpacity
            style={styles.pickerContainer}
            onPress={() => setShowClassPicker(true)}
          >
            <Text style={styles.pickerText}>
              {selectedClassData ? `${selectedClassData.name} - ${selectedClassData.batch} ` : 'Select a class'}
            </Text>
            <ChevronDown color="#4b5563" size={20} />
          </TouchableOpacity>

          {selectedClassData && (
            <View style={styles.statsRow}>
              <View style={styles.miniStat}>
                <Text style={styles.miniStatLabel}>BATCH</Text>
                <Text style={styles.miniStatValue}>{selectedClassData.batch}</Text>
              </View>
              <View style={styles.miniStat}>
                <Text style={styles.miniStatLabel}>TOTAL</Text>
                <Text style={styles.miniStatValue}>{selectedClassData.total_students}</Text>
              </View>
            </View>
          )}
        </View>

        {/* Today's Summary */}
        {todaySummary && (
          <View style={styles.card}>
            <Text style={styles.sectionTitle}>Today's Summary</Text>
            <View style={styles.statsRow}>
              <View style={[styles.miniStat, { backgroundColor: '#dcfce7' }]}>
                <Text style={[styles.miniStatLabel, { color: '#16a34a' }]}>PRESENT</Text>
                <Text style={[styles.miniStatValue, { color: '#16a34a' }]}>{todaySummary.present}</Text>
              </View>
              <View style={[styles.miniStat, { backgroundColor: '#fee2e2' }]}>
                <Text style={[styles.miniStatLabel, { color: '#dc2626' }]}>ABSENT</Text>
                <Text style={[styles.miniStatValue, { color: '#dc2626' }]}>{todaySummary.absent}</Text>
              </View>
              <View style={[styles.miniStat, { backgroundColor: '#dbeafe' }]}>
                <Text style={[styles.miniStatLabel, { color: '#2563eb' }]}>PERCENTAGE</Text>
                <Text style={[styles.miniStatValue, { color: '#2563eb' }]}>{todaySummary.percentage}%</Text>
              </View>
            </View>
          </View>
        )}

        {/* Action Buttons */}
        <View style={styles.actionGrid}>
          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: '#16a34a' }]}
            onPress={handleStartAttendance}
          >
            <Camera color="white" size={32} style={{ marginBottom: 8 }} />
            <Text style={styles.actionButtonText}>Start{'\n'}Attendance</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: '#6366f1' }]}
            onPress={() => {
              setSelectedClass(selectedClassData);
              navigateTo('DetailedReport');
            }}
          >
            <BarChart3 color="white" size={32} style={{ marginBottom: 8 }} />
            <Text style={styles.actionButtonText}>View{'\n'}Reports</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: '#ea580c' }]}
            onPress={() => navigateTo('RegisterStudent')}
          >
            <Camera color="white" size={32} style={{ marginBottom: 8 }} />
            <Text style={styles.actionButtonText}>Register{'\n'}Student</Text>
          </TouchableOpacity>
        </View>

        {/* Defaulters List */}
        {defaulters.length > 0 && (
          <View style={styles.card}>
            <View style={styles.cardHeader}>
              <Text style={styles.sectionTitle}>Defaulter List (Top 5)</Text>
              <TouchableOpacity onPress={() => navigateTo('NotificationHub')}>
                <Text style={styles.linkText}>Notify All</Text>
              </TouchableOpacity>
            </View>

            {defaulters.slice(0, 5).map((student, index) => (
              <View key={index} style={styles.reportRow}>
                <Text style={styles.reportLabel}>{student.name}</Text>
                <Text style={[styles.reportValue, { color: '#dc2626', backgroundColor: '#fee2e2' }]}>
                  {student.attendance}%
                </Text>
              </View>
            ))}

            <TouchableOpacity
              style={styles.outlineButton}
              onPress={() => navigateTo('NotificationHub')}
            >
              <Text style={styles.outlineButtonText}>🚨 Send Notifications</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>

      {/* Bottom Nav */}
      <View style={styles.bottomNav}>
        <View style={styles.navItem}>
          <Home color="#2563eb" size={24} />
          <Text style={[styles.navText, { color: '#2563eb' }]}>Home</Text>
        </View>
        <TouchableOpacity style={styles.navItem} onPress={() => navigateTo('DetailedReport')}>
          <BarChart3 color="#9ca3af" size={24} />
          <Text style={styles.navText}>Reports</Text>
        </TouchableOpacity>
        <View style={styles.navItem}>
          <Settings color="#9ca3af" size={24} />
          <Text style={styles.navText}>Settings</Text>
        </View>
      </View>
    </View>
  );
};

// --- 4. REGISTER STUDENT SCREEN ---
const RegisterStudentScreen = ({ navigateTo }) => {
  const [permission, requestPermission] = useCameraPermissions();
  const [name, setName] = useState('');
  const [rollNo, setRollNo] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [department, setDepartment] = useState('');
  const [batch, setBatch] = useState('');
  const [photo, setPhoto] = useState(null);
  const [loading, setLoading] = useState(false);
  const [facing, setFacing] = useState('front');
  const cameraRef = useRef(null);

  if (!permission) return <View />;
  if (!permission.granted) {
    return (
      <View style={[styles.screenContainer, { justifyContent: 'center', alignItems: 'center' }]}>
        <Text>Camera permission required</Text>
        <TouchableOpacity style={styles.primaryButton} onPress={requestPermission}>
          <Text style={styles.primaryButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const toggleCameraFacing = () => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  };

  const takePicture = async () => {
    if (cameraRef.current) {
      try {
        const photoData = await cameraRef.current.takePictureAsync({
          quality: 0.8,
          base64: false,
        });
        setPhoto(photoData.uri);
      } catch (error) {
        Alert.alert("Error", "Failed to capture image");
      }
    }
  };

  const handleRegister = async () => {
    if (!name || !rollNo || !department || !photo) {
      Alert.alert('Error', 'Please fill required fields and capture photo');
      return;
    }

    setLoading(true);
    try {
      // First, upload photo to get face encoding
      const formData = new FormData();
      formData.append('file', {
        uri: photo,
        type: 'image/jpeg',
        name: 'student.jpg',
      });

      // Get auth headers
      const token = await AsyncStorage.getItem('userToken');
      const headers = {
        'Authorization': `Bearer ${token}`
      };

      const response = await fetch(`${api.BASE_URL}/scan`, {
        method: 'POST',
        body: formData,
        headers: headers,
      });

      const scanResult = await response.json();

      if (scanResult.status === 'error' || !scanResult.people || scanResult.people.length === 0) {
        Alert.alert('Error', 'No face detected in photo. Please try again.');
        setPhoto(null);
        return;
      }

      // Get the face encoding from scan result
      const encoding = scanResult.people[0].encoding || [];

      if (encoding.length !== 128) {
        Alert.alert('Error', 'Invalid face encoding. Please retake photo.');
        setPhoto(null);
        return;
      }

      // Register student with encoding
      const result = await api.registerStudent(
        name,
        rollNo,
        encoding,
        email,
        phone,
        department,
        batch,
        null // user_id - optional
      );

      Alert.alert('Success', 'Student registered successfully!', [
        { text: 'OK', onPress: () => navigateTo('TeacherDashboard') }
      ]);
    } catch (error) {
      Alert.alert('Registration Failed', error.message || 'Could not register student');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.screenContainer}>
      <View style={styles.header}>
        <TouchableOpacity
          onPress={() => navigateTo('TeacherDashboard')}
          style={{ padding: 10, marginRight: 8 }}
        >
          <ChevronLeft color="#374151" size={30} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Register New Student</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView style={styles.scrollContent}>
        {/* Camera Preview */}
        <View style={[styles.cameraContainer, { height: 300, marginBottom: 20 }]}>
          {photo ? (
            <Image source={{ uri: photo }} style={styles.preview} />
          ) : (
            <CameraView style={styles.camera} facing={facing} ref={cameraRef}>
              <View style={styles.overlay}>
                <Text style={styles.overlayText}>[ Face Here ]</Text>
              </View>
              <TouchableOpacity style={styles.flipBtn} onPress={toggleCameraFacing}>
                <Text style={{ color: 'white' }}>Flip</Text>
              </TouchableOpacity>
            </CameraView>
          )}
        </View>

        {!photo ? (
          <TouchableOpacity style={styles.primaryButton} onPress={takePicture}>
            <Text style={styles.primaryButtonText}>📸 Capture Student Photo</Text>
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={[styles.primaryButton, { backgroundColor: '#6b7280' }]}
            onPress={() => setPhoto(null)}
          >
            <Text style={styles.primaryButtonText}>↻ Retake Photo</Text>
          </TouchableOpacity>
        )}

        <View style={styles.card}>
          <Text style={styles.cardLabel}>STUDENT INFORMATION</Text>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Full Name *</Text>
            <TextInput
              style={styles.input}
              placeholder="Enter student name"
              placeholderTextColor="#9ca3af"
              value={name}
              onChangeText={setName}
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Roll Number *</Text>
            <TextInput
              style={styles.input}
              placeholder="e.g., MCA001"
              placeholderTextColor="#9ca3af"
              value={rollNo}
              onChangeText={setRollNo}
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Department *</Text>
            <TextInput
              style={styles.input}
              placeholder="e.g., MCA"
              placeholderTextColor="#9ca3af"
              value={department}
              onChangeText={setDepartment}
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Batch</Text>
            <TextInput
              style={styles.input}
              placeholder="e.g., MCA 2A"
              placeholderTextColor="#9ca3af"
              value={batch}
              onChangeText={setBatch}
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Email</Text>
            <TextInput
              style={styles.input}
              placeholder="student@institute.edu"
              placeholderTextColor="#9ca3af"
              value={email}
              onChangeText={setEmail}
              autoCapitalize="none"
              keyboardType="email-address"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Phone</Text>
            <TextInput
              style={styles.input}
              placeholder="+91 9876543210"
              placeholderTextColor="#9ca3af"
              value={phone}
              onChangeText={setPhone}
              keyboardType="phone-pad"
            />
          </View>

          <TouchableOpacity
            style={styles.primaryButton}
            onPress={handleRegister}
            disabled={loading}
          >
            {loading ? (
              <ActivityIndicator color="white" />
            ) : (
              <Text style={styles.primaryButtonText}>✓ Register Student</Text>
            )}
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
};

// --- 5. SCAN ATTENDANCE (WITH BACKEND LOGIC) ---
const ScanAttendanceScreen = ({ navigateTo, currentSession }) => {
  const [permission, requestPermission] = useCameraPermissions();
  const [photo, setPhoto] = useState(null);
  const [status, setStatus] = useState("Searching for Faces...");
  const [scanResult, setScanResult] = useState(null);
  const [facing, setFacing] = useState("back");
  const [scannedCount, setScannedCount] = useState(0);
  const cameraRef = useRef(null);

  if (!permission) return <View />;
  if (!permission.granted) {
    return (
      <View style={[styles.screenContainer, { justifyContent: 'center', alignItems: 'center' }]}>
        <Text>Camera permission required</Text>
        <TouchableOpacity style={styles.primaryButton} onPress={requestPermission}>
          <Text style={styles.primaryButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const toggleCameraFacing = () => {
    setFacing(current => (current === 'back' ? 'front' : 'back'));
  };

  const takePicture = async () => {
    if (cameraRef.current) {
      try {
        const photoData = await cameraRef.current.takePictureAsync({
          quality: 0.5,
          base64: false,
        });
        setPhoto(photoData.uri);
        uploadImage(photoData.uri);
      } catch (error) {
        Alert.alert("Error", "Failed to capture image");
      }
    }
  };

  const uploadImage = async (imageUri) => {
    setStatus("Analyzing...");
    const formData = new FormData();
    formData.append('file', {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'scan.jpg',
    });
    formData.append('session_id', currentSession);

    try {
      const response = await fetch(`${api.BASE_URL}/scan`, {
        method: 'POST',
        body: formData,
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      const result = await response.json();

      if (result.status === 'error') {
        setScanResult([{ name: "No Face Detected", status: 'Retry' }]);
        setStatus(result.message || "No Face Detected");
        return;
      }

      if (result.people && result.people.length > 0) {
        // Process all detected people
        const detectedPeople = result.people.map(person => {
          if (person.name === "Unknown") {
            return { name: "Unknown Face", status: 'Absent' };
          } else {
            return { name: person.name, status: 'Present' };
          }
        });

        setScanResult(detectedPeople);

        // Count how many were successfully identified
        const identifiedCount = detectedPeople.filter(p => p.status === 'Present').length;
        const unknownCount = detectedPeople.filter(p => p.status === 'Absent').length;

        if (identifiedCount > 0) {
          setStatus(`${identifiedCount} Student${identifiedCount > 1 ? 's' : ''} Verified`);
          setScannedCount(prev => prev + identifiedCount);
        } else {
          setStatus("No Match Found");
        }
      }

    } catch (error) {
      setStatus("Connection Error");
      Alert.alert("Error", "Check backend server");
    }
  };

  const handleStopSession = async () => {
    try {
      await api.stopSession(currentSession);
      Alert.alert("Success", `Session stopped. ${scannedCount} students scanned.`, [
        { text: "OK", onPress: () => navigateTo('TeacherDashboard') }
      ]);
    } catch (error) {
      Alert.alert("Error", "Failed to stop session");
    }
  };

  return (
    <View style={[styles.screenContainer, { backgroundColor: '#111827' }]}>
      <View style={[styles.header, { backgroundColor: '#1f2937', borderBottomWidth: 0 }]}>
        <TouchableOpacity
          onPress={() => navigateTo('TeacherDashboard')}
          style={{ padding: 10, marginRight: 8 }}
        >
          <ChevronLeft color="white" size={30} />
        </TouchableOpacity>
        <View style={{ alignItems: 'center', flex: 1 }}>
          <Text style={[styles.headerTitle, { color: 'white' }]}>Scanning Attendance</Text>
          <Text style={{ color: '#9ca3af', fontSize: 10 }}>{status} • Scanned: {scannedCount}</Text>
        </View>
        <View style={styles.liveBadge}>
          <Text style={styles.liveText}>LIVE</Text>
        </View>
      </View>

      <View style={styles.cameraContainer}>
        {photo ? (
          <Image source={{ uri: photo }} style={styles.preview} />
        ) : (
          <CameraView style={styles.camera} facing={facing} ref={cameraRef}>
            <View style={styles.overlay}>
              <Text style={styles.overlayText}>[ Face Here ]</Text>
            </View>
            <TouchableOpacity style={styles.flipBtn} onPress={toggleCameraFacing}>
              <Text style={{ color: 'white' }}>Flip</Text>
            </TouchableOpacity>
          </CameraView>
        )}
      </View>

      {/* Result Cards */}
      <View style={{ padding: 16 }}>
        {scanResult && Array.isArray(scanResult) && scanResult.length > 0 && (
          <>
            {scanResult.map((person, index) => (
              <View
                key={index}
                style={[
                  styles.resultCard,
                  person.status === 'Present' ? styles.bgGreen : styles.bgRed,
                  { marginBottom: 8 }
                ]}
              >
                <Text style={styles.resultText}>{person.name}</Text>
                <View style={styles.resultBadge}>
                  <Text style={{ fontWeight: 'bold', color: 'white' }}>
                    {person.status === 'Present' ? '✅ Present' : (person.status === 'Retry' ? '❌ Retry' : '❓ Absent')}
                  </Text>
                </View>
              </View>
            ))}
          </>
        )}

        <View style={{ flexDirection: 'row', gap: 8 }}>
          {photo ? (
            <TouchableOpacity
              style={[styles.primaryButton, { flex: 1 }]}
              onPress={() => { setPhoto(null); setScanResult(null); setStatus("Searching..."); }}
            >
              <Text style={styles.primaryButtonText}>Scan Next</Text>
            </TouchableOpacity>
          ) : (
            <TouchableOpacity style={[styles.primaryButton, { flex: 1 }]} onPress={takePicture}>
              <Text style={styles.primaryButtonText}>Capture & Check</Text>
            </TouchableOpacity>
          )}

          <TouchableOpacity
            style={[styles.primaryButton, { flex: 1, backgroundColor: '#dc2626' }]}
            onPress={handleStopSession}
          >
            <Text style={styles.primaryButtonText}>Stop Session</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
};

// --- 4. STUDENT DASHBOARD (UPGRADED) ---
const StudentDashboard = ({ navigateTo, userInfo, onLogout }) => {
  const [studentReport, setStudentReport] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStudentData();
  }, []);

  const loadStudentData = async () => {
    try {
      const result = await api.getStudentReport();
      if (result.status === 'success') {
        setStudentReport(result.data);
      }
    } catch (error) {
      console.error('Error loading student data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <View style={[styles.screenContainer, { justifyContent: 'center', alignItems: 'center' }]}>
        <ActivityIndicator size="large" color="#2563eb" />
      </View>
    );
  }

  const overallPercentage = studentReport?.overall_percentage || 0;
  const isDefaulter = overallPercentage < 75;

  return (
    <View style={styles.screenContainer}>
      {/* Header */}
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>My Attendance</Text>
          <Text style={styles.headerSubtitle}>Welcome, {userInfo?.name}</Text>
        </View>
        <TouchableOpacity onPress={onLogout}>
          <LogOut color="#4b5563" size={24} />
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scrollContent}>
        {/* Profile Card */}
        <View style={styles.profileCard}>
          <View style={styles.avatar}>
            <Text style={{ color: 'white', fontSize: 22 }}>👤</Text>
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.profileName}>{userInfo?.name}</Text>
            <Text style={styles.profileDetail}>
              Roll No: {userInfo?.roll_no || 'N/A'} | {userInfo?.department || 'N/A'}
            </Text>
          </View>
        </View>

        {/* Overall Stats */}
        <View style={styles.card}>
          <Text style={styles.cardLabel}>OVERALL PERFORMANCE</Text>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 8 }}>
            <Text style={styles.label}>Attendance Percentage</Text>
            <Text style={[styles.statValue, { color: isDefaulter ? '#dc2626' : '#16a34a' }]}>
              {overallPercentage}%
            </Text>
          </View>
          <View style={styles.progressBarBg}>
            <View style={[
              styles.progressBarFill,
              { width: `${Math.min(overallPercentage, 100)}%`, backgroundColor: isDefaulter ? '#dc2626' : '#16a34a' }
            ]} />
          </View>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginTop: 8 }}>
            <Text style={{ fontSize: 12, color: '#6b7280' }}>Present: {studentReport?.total_present || 0}</Text>
            <Text style={{ fontSize: 12, color: '#6b7280' }}>Total: {studentReport?.total_classes || 0}</Text>
          </View>
          {isDefaulter && (
            <View style={styles.warningBadge}>
              <Text style={styles.warningText}>⚠ Below required minimum (75%)</Text>
            </View>
          )}
        </View>

        {/* Quick Action Cards */}
        <Text style={[styles.sectionTitle, { marginBottom: 8 }]}>Quick Actions</Text>
        <View style={{ flexDirection: 'row', gap: 10, marginBottom: 16 }}>
          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: '#2563eb', flex: 1 }]}
            onPress={() => navigateTo('StudentTimeline')}
          >
            <Calendar color="white" size={26} style={{ marginBottom: 6 }} />
            <Text style={styles.actionButtonText}>Attendance{`\n`}Calendar</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: '#7c3aed', flex: 1 }]}
            onPress={() => navigateTo('StudentSessions')}
          >
            <BarChart3 color="white" size={26} style={{ marginBottom: 6 }} />
            <Text style={styles.actionButtonText}>Session{`\n`}History</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: '#0891b2', flex: 1 }]}
            onPress={() => navigateTo('StudentProfile')}
          >
            <Settings color="white" size={26} style={{ marginBottom: 6 }} />
            <Text style={styles.actionButtonText}>My{`\n`}Profile</Text>
          </TouchableOpacity>
        </View>

        {/* Subject Breakdown */}
        <Text style={styles.sectionTitle}>Subject Breakdown</Text>
        {(!studentReport?.subjects || studentReport.subjects.length === 0) && (
          <View style={[styles.card, { alignItems: 'center', padding: 24 }]}>
            <Text style={{ color: '#6b7280', fontSize: 14 }}>No attendance data yet.</Text>
          </View>
        )}
        {studentReport?.subjects?.map((subject, index) => (
          <View key={index} style={styles.subjectCard}>
            <View style={{ flex: 1 }}>
              <Text style={styles.subjectTitle}>{subject.name}</Text>
              <Text style={styles.subjectDetail}>
                {subject.present}/{subject.total_classes} classes attended
              </Text>
            </View>
            <View style={{ alignItems: 'flex-end' }}>
              <Text style={[styles.subjectValue, { color: subject.percentage >= 75 ? '#16a34a' : '#dc2626' }]}>
                {subject.percentage}%
              </Text>
              <Text style={[
                styles.statusTag,
                {
                  backgroundColor: subject.percentage >= 75 ? '#dcfce7' : '#fee2e2',
                  color: subject.percentage >= 75 ? '#16a34a' : '#dc2626'
                }
              ]}>
                {subject.status}
              </Text>
            </View>
          </View>
        ))}
      </ScrollView>
    </View>
  );
};

// --- STUDENT TIMELINE SCREEN ---
const StudentTimelineScreen = ({ navigateTo }) => {
  const [month, setMonth] = useState(new Date().getMonth() + 1);
  const [year, setYear] = useState(new Date().getFullYear());
  const [days, setDays] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => { loadTimeline(); }, [month, year]);

  const loadTimeline = async () => {
    setLoading(true);
    try {
      const res = await api.getStudentTimeline(month, year);
      if (res.status === 'success') setDays(res.data);
    } catch (e) {
      Alert.alert('Error', 'Could not load timeline');
    } finally {
      setLoading(false);
    }
  };

  // Build lookup: 'YYYY-MM-DD' → entry
  const dayMap = {};
  days.forEach(d => {
    if (!dayMap[d.date]) dayMap[d.date] = [];
    dayMap[d.date].push(d);
  });

  const daysInMonth = new Date(year, month, 0).getDate();
  const firstDayOfWeek = new Date(year, month - 1, 1).getDay();
  const cells = [
    ...Array.from({ length: firstDayOfWeek }, () => null),
    ...Array.from({ length: daysInMonth }, (_, i) => i + 1),
  ];

  const monthLabel = new Date(year, month - 1).toLocaleString('default', { month: 'long', year: 'numeric' });

  const prevMonth = () => {
    if (month === 1) { setMonth(12); setYear(y => y - 1); }
    else setMonth(m => m - 1);
  };
  const nextMonth = () => {
    if (month === 12) { setMonth(1); setYear(y => y + 1); }
    else setMonth(m => m + 1);
  };

  const cellKey = (day) =>
    `${year}-${String(month).padStart(2, '0')}-${String(day).padStart(2, '0')}`;

  const getCellBg = (day) => {
    const entries = dayMap[cellKey(day)];
    if (!entries || entries.length === 0) return '#f3f4f6';
    const hasPresent = entries.some(e => e.status === 'Present');
    return hasPresent ? '#dcfce7' : '#fee2e2';
  };

  const getCellIcon = (day) => {
    const entries = dayMap[cellKey(day)];
    if (!entries || entries.length === 0) return '';
    const hasPresent = entries.some(e => e.status === 'Present');
    return hasPresent ? '✓' : '✗';
  };

  return (
    <View style={styles.screenContainer}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigateTo('StudentDashboard')} style={{ padding: 10 }}>
          <ChevronLeft color="#374151" size={28} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Attendance Calendar</Text>
        <View style={{ width: 40 }} />
      </View>

      {/* Month Navigation */}
      <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', paddingHorizontal: 16, paddingVertical: 12 }}>
        <TouchableOpacity onPress={prevMonth} style={{ padding: 8 }}>
          <ChevronLeft color="#2563eb" size={24} />
        </TouchableOpacity>
        <Text style={{ fontSize: 16, fontWeight: '700', color: '#111827' }}>{monthLabel}</Text>
        <TouchableOpacity onPress={nextMonth} style={{ padding: 8 }}>
          <Text style={{ color: '#2563eb', fontSize: 22, fontWeight: '600' }}>›</Text>
        </TouchableOpacity>
      </View>

      {/* Day Labels */}
      <View style={{ flexDirection: 'row', paddingHorizontal: 16, marginBottom: 4 }}>
        {['Su', 'Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa'].map(d => (
          <Text key={d} style={{ flex: 1, textAlign: 'center', fontSize: 11, color: '#6b7280', fontWeight: '700' }}>{d}</Text>
        ))}
      </View>

      {loading ? (
        <ActivityIndicator size="large" color="#2563eb" style={{ marginTop: 40 }} />
      ) : (
        <ScrollView style={{ paddingHorizontal: 12 }}>
          {/* Calendar Grid */}
          <View style={{ flexDirection: 'row', flexWrap: 'wrap' }}>
            {cells.map((day, idx) => (
              <TouchableOpacity
                key={idx}
                style={{ width: '14.28%', aspectRatio: 1, padding: 2 }}
                onPress={() => day && dayMap[cellKey(day)] && setSelected({ key: cellKey(day), entries: dayMap[cellKey(day)] })}
                disabled={!day}
              >
                <View style={{
                  flex: 1,
                  borderRadius: 8,
                  backgroundColor: day ? getCellBg(day) : 'transparent',
                  justifyContent: 'center',
                  alignItems: 'center',
                  borderWidth: day ? 1 : 0,
                  borderColor: '#e5e7eb',
                }}>
                  {day && <Text style={{ fontSize: 12, fontWeight: '600', color: '#111827' }}>{day}</Text>}
                  {day && <Text style={{ fontSize: 10 }}>{getCellIcon(day)}</Text>}
                </View>
              </TouchableOpacity>
            ))}
          </View>

          {/* Legend */}
          <View style={{ flexDirection: 'row', gap: 16, padding: 16, justifyContent: 'center' }}>
            {[['#dcfce7', 'Present'], ['#fee2e2', 'Absent'], ['#f3f4f6', 'No Class']].map(([color, label]) => (
              <View key={label} style={{ flexDirection: 'row', alignItems: 'center', gap: 6 }}>
                <View style={{ width: 12, height: 12, borderRadius: 3, backgroundColor: color, borderWidth: 1, borderColor: '#e5e7eb' }} />
                <Text style={{ fontSize: 12, color: '#6b7280' }}>{label}</Text>
              </View>
            ))}
          </View>

          {/* Monthly Summary */}
          {days.length > 0 && (
            <View style={styles.card}>
              <Text style={styles.cardLabel}>THIS MONTH</Text>
              <View style={{ flexDirection: 'row', justifyContent: 'space-around' }}>
                <View style={{ alignItems: 'center' }}>
                  <Text style={{ fontSize: 24, fontWeight: '700', color: '#16a34a' }}>
                    {days.filter(d => d.status === 'Present').length}
                  </Text>
                  <Text style={{ fontSize: 12, color: '#6b7280' }}>Present</Text>
                </View>
                <View style={{ alignItems: 'center' }}>
                  <Text style={{ fontSize: 24, fontWeight: '700', color: '#dc2626' }}>
                    {days.filter(d => d.status === 'Absent').length}
                  </Text>
                  <Text style={{ fontSize: 12, color: '#6b7280' }}>Absent</Text>
                </View>
                <View style={{ alignItems: 'center' }}>
                  <Text style={{ fontSize: 24, fontWeight: '700', color: '#2563eb' }}>
                    {days.length > 0 ? Math.round((days.filter(d => d.status === 'Present').length / days.length) * 100) : 0}%
                  </Text>
                  <Text style={{ fontSize: 12, color: '#6b7280' }}>This Month</Text>
                </View>
              </View>
            </View>
          )}
        </ScrollView>
      )}

      {/* Day Detail Modal */}
      <Modal visible={!!selected} transparent animationType="slide" onRequestClose={() => setSelected(null)}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>{selected?.key}</Text>
            {selected?.entries?.map((entry, i) => (
              <View key={i} style={[styles.reportRow, { marginBottom: 8 }]}>
                <View>
                  <Text style={{ fontWeight: '600', color: '#111827' }}>{entry.subject}</Text>
                  <Text style={{ fontSize: 12, color: '#6b7280' }}>Time: {entry.time} | By: {entry.marked_by}</Text>
                  {entry.faculty_name && (
                    <Text style={{ fontSize: 12, color: '#6b7280' }}>Faculty: {entry.faculty_name}</Text>
                  )}
                </View>
                <Text style={[
                  styles.reportValue,
                  { color: entry.status === 'Present' ? '#16a34a' : '#dc2626',
                    backgroundColor: entry.status === 'Present' ? '#dcfce7' : '#fee2e2' }
                ]}>
                  {entry.status}
                </Text>
              </View>
            ))}
            <TouchableOpacity style={styles.modalCloseButton} onPress={() => setSelected(null)}>
              <Text style={styles.modalCloseButtonText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
};

// --- STUDENT SESSION HISTORY SCREEN ---
const StudentSessionsScreen = ({ navigateTo }) => {
  const [sessions, setSessions] = useState([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const LIMIT = 15;

  useEffect(() => { loadSessions(1, true); }, []);

  const loadSessions = async (pageNum, reset = false) => {
    if (reset) setLoading(true); else setLoadingMore(true);
    try {
      const res = await api.getStudentSessions(pageNum, LIMIT);
      if (res.status === 'success') {
        const newData = res.data.sessions || [];
        setSessions(prev => reset ? newData : [...prev, ...newData]);
        setTotal(res.data.total || 0);
        setPage(pageNum);
      }
    } catch (e) {
      Alert.alert('Error', 'Could not load session history');
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  };

  const hasMore = sessions.length < total;

  const getStatusStyle = (status) => ({
    color: status === 'Present' ? '#16a34a' : '#dc2626',
    backgroundColor: status === 'Present' ? '#dcfce7' : '#fee2e2',
  });

  return (
    <View style={styles.screenContainer}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigateTo('StudentDashboard')} style={{ padding: 10 }}>
          <ChevronLeft color="#374151" size={28} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Session History</Text>
        <Text style={{ color: '#6b7280', fontSize: 13, marginRight: 4 }}>{total} total</Text>
      </View>

      {loading ? (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
          <ActivityIndicator size="large" color="#2563eb" />
        </View>
      ) : sessions.length === 0 ? (
        <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center', padding: 40 }}>
          <Text style={{ color: '#6b7280', fontSize: 16, textAlign: 'center' }}>No attendance records yet.</Text>
        </View>
      ) : (
        <ScrollView style={styles.scrollContent}>
          {/* Column Headers */}
          <View style={[styles.reportRow, { backgroundColor: '#f3f4f6', borderRadius: 8, marginBottom: 4 }]}>
            <Text style={{ flex: 2, fontWeight: '700', fontSize: 12, color: '#374151' }}>DATE / SUBJECT</Text>
            <Text style={{ fontWeight: '700', fontSize: 12, color: '#374151', marginRight: 8 }}>STATUS</Text>
            <Text style={{ fontWeight: '700', fontSize: 12, color: '#374151' }}>BY</Text>
          </View>

          {sessions.map((s, idx) => (
            <View key={s.log_id || idx} style={[styles.reportRow, { paddingVertical: 10, borderBottomWidth: 1, borderBottomColor: '#f3f4f6' }]}>
              <View style={{ flex: 2 }}>
                <Text style={{ fontWeight: '600', color: '#111827', fontSize: 14 }}>{s.subject}</Text>
                <Text style={{ fontSize: 12, color: '#6b7280' }}>{s.date} · {s.time}</Text>
              </View>
              <Text style={[styles.reportValue, getStatusStyle(s.status), { marginRight: 6 }]}>
                {s.status}
              </Text>
              <Text style={[
                styles.statusTag,
                {
                  backgroundColor: s.marked_by === 'AI' ? '#ede9fe' : '#fef3c7',
                  color: s.marked_by === 'AI' ? '#7c3aed' : '#92400e',
                }
              ]}>
                {s.marked_by}
              </Text>
            </View>
          ))}

          {hasMore && (
            <TouchableOpacity
              style={[styles.outlineButton, { marginVertical: 16 }]}
              onPress={() => loadSessions(page + 1)}
              disabled={loadingMore}
            >
              {loadingMore
                ? <ActivityIndicator color="#2563eb" />
                : <Text style={styles.outlineButtonText}>Load More ({total - sessions.length} remaining)</Text>
              }
            </TouchableOpacity>
          )}
        </ScrollView>
      )}
    </View>
  );
};

// --- STUDENT PROFILE SCREEN ---
const StudentProfileScreen = ({ navigateTo, userInfo, setUserInfo }) => {
  const [name, setName] = useState(userInfo?.name || '');
  const [phone, setPhone] = useState('');
  const [department, setDepartment] = useState(userInfo?.department || '');
  const [rollNo, setRollNo] = useState(userInfo?.roll_no || '');
  const [batch, setBatch] = useState(userInfo?.batch || '');
  const [email, setEmail] = useState(userInfo?.email || '');
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [isEditing, setIsEditing] = useState(false);

  useEffect(() => { fetchProfile(); }, []);

  const fetchProfile = async () => {
    try {
      const res = await api.getStudentProfile();
      if (res.status === 'success') {
        const p = res.data;
        setName(p.name || '');
        setPhone(p.phone || '');
        setDepartment(p.department || '');
        setRollNo(p.roll_no || '');
        setBatch(p.batch || '');
        setEmail(p.email || '');
      }
    } catch (e) {
      Alert.alert('Error', 'Could not load profile');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!name.trim()) {
      Alert.alert('Error', 'Name cannot be empty');
      return;
    }
    setSaving(true);
    try {
      const res = await api.updateStudentProfile({ name, phone, department });
      if (res.status === 'success') {
        if (setUserInfo) setUserInfo(prev => ({ ...prev, name, department }));
        setIsEditing(false);
        Alert.alert('Success', 'Profile updated successfully!');
      } else {
        Alert.alert('Error', res.message || 'Update failed');
      }
    } catch (e) {
      Alert.alert('Error', 'Could not save profile');
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <View style={[styles.screenContainer, { justifyContent: 'center', alignItems: 'center' }]}>
        <ActivityIndicator size="large" color="#2563eb" />
      </View>
    );
  }

  return (
    <View style={styles.screenContainer}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigateTo('StudentDashboard')} style={{ padding: 10 }}>
          <ChevronLeft color="#374151" size={28} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>My Profile</Text>
        <TouchableOpacity
          onPress={() => setIsEditing(prev => !prev)}
          style={{ padding: 10 }}
        >
          <Text style={{ color: '#2563eb', fontWeight: '600', fontSize: 15 }}>
            {isEditing ? 'Cancel' : 'Edit'}
          </Text>
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scrollContent}>
        {/* Avatar */}
        <View style={{ alignItems: 'center', paddingVertical: 24 }}>
          <View style={[styles.avatar, { width: 72, height: 72, borderRadius: 36 }]}>
            <Text style={{ color: 'white', fontSize: 30 }}>👤</Text>
          </View>
          <Text style={[styles.profileName, { marginTop: 8 }]}>{name}</Text>
          <Text style={styles.profileDetail}>Roll No: {rollNo || 'N/A'}{batch ? ` · ${batch}` : ''}</Text>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardLabel}>{isEditing ? 'EDIT INFORMATION' : 'PROFILE INFORMATION'}</Text>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Full Name</Text>
            {isEditing ? (
              <TextInput
                style={styles.input}
                value={name}
                onChangeText={setName}
                placeholder="Your full name"
                placeholderTextColor="#9ca3af"
              />
            ) : (
              <Text style={styles.profileValueText}>{name || '—'}</Text>
            )}
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Phone</Text>
            {isEditing ? (
              <TextInput
                style={styles.input}
                value={phone}
                onChangeText={setPhone}
                placeholder="+91 9876543210"
                placeholderTextColor="#9ca3af"
                keyboardType="phone-pad"
              />
            ) : (
              <Text style={styles.profileValueText}>{phone || '—'}</Text>
            )}
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Department</Text>
            {isEditing ? (
              <TextInput
                style={styles.input}
                value={department}
                onChangeText={setDepartment}
                placeholder="e.g., MCA"
                placeholderTextColor="#9ca3af"
              />
            ) : (
              <Text style={styles.profileValueText}>{department || '—'}</Text>
            )}
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardLabel}>READ-ONLY INFORMATION</Text>
          <View style={styles.reportRow}>
            <Text style={styles.label}>Email</Text>
            <Text style={{ color: '#374151', fontSize: 14 }}>{email || userInfo?.email || '—'}</Text>
          </View>
          <View style={styles.reportRow}>
            <Text style={styles.label}>Roll Number</Text>
            <Text style={{ color: '#374151', fontSize: 14 }}>{rollNo || '—'}</Text>
          </View>
          <View style={styles.reportRow}>
            <Text style={styles.label}>Batch</Text>
            <Text style={{ color: '#374151', fontSize: 14 }}>{batch || '—'}</Text>
          </View>
        </View>

        {isEditing && (
          <TouchableOpacity
            style={styles.primaryButton}
            onPress={handleSave}
            disabled={saving}
          >
            {saving ? <ActivityIndicator color="white" /> : <Text style={styles.primaryButtonText}>✓ Save Changes</Text>}
          </TouchableOpacity>
        )}

        <TouchableOpacity
          style={[styles.outlineButton, { marginTop: isEditing ? 12 : 0 }]}
          onPress={() => navigateTo('ChangePassword')}
        >
          <Text style={styles.outlineButtonText}>🔒 Change Password</Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
};

// --- CHANGE PASSWORD SCREEN ---
const ChangePasswordScreen = ({ navigateTo }) => {
  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [showOld, setShowOld] = useState(false);
  const [showNew, setShowNew] = useState(false);

  const handleChangePassword = async () => {
    if (!oldPassword || !newPassword || !confirmPassword) {
      Alert.alert('Error', 'Please fill all fields');
      return;
    }
    if (newPassword.length < 8) {
      Alert.alert('Error', 'New password must be at least 8 characters');
      return;
    }
    if (newPassword !== confirmPassword) {
      Alert.alert('Error', 'New passwords do not match');
      return;
    }
    if (oldPassword === newPassword) {
      Alert.alert('Error', 'New password must be different from current password');
      return;
    }

    setLoading(true);
    try {
      const res = await api.changePassword(oldPassword, newPassword);
      if (res.status === 'success') {
        Alert.alert('Success', 'Password changed successfully!', [
          { text: 'OK', onPress: () => navigateTo('StudentProfile') }
        ]);
        setOldPassword('');
        setNewPassword('');
        setConfirmPassword('');
      } else {
        Alert.alert('Error', res.message || 'Could not change password');
      }
    } catch (e) {
      Alert.alert('Error', 'Network error. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.screenContainer}>
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigateTo('StudentProfile')} style={{ padding: 10 }}>
          <ChevronLeft color="#374151" size={28} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Change Password</Text>
        <View style={{ width: 40 }} />
      </View>

      <ScrollView style={styles.scrollContent}>
        <View style={styles.card}>
          <Text style={[styles.label, { marginBottom: 16, color: '#6b7280' }]}>
            Choose a strong password with at least 8 characters.
          </Text>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Current Password</Text>
            <View style={{ position: 'relative' }}>
              <TextInput
                style={styles.input}
                value={oldPassword}
                onChangeText={setOldPassword}
                secureTextEntry={!showOld}
                placeholder="Enter current password"
                placeholderTextColor="#9ca3af"
              />
              <TouchableOpacity
                onPress={() => setShowOld(v => !v)}
                style={{ position: 'absolute', right: 12, top: 12 }}
              >
                <Text style={{ color: '#6b7280', fontSize: 13 }}>{showOld ? 'Hide' : 'Show'}</Text>
              </TouchableOpacity>
            </View>
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>New Password</Text>
            <View style={{ position: 'relative' }}>
              <TextInput
                style={styles.input}
                value={newPassword}
                onChangeText={setNewPassword}
                secureTextEntry={!showNew}
                placeholder="Min. 8 characters"
                placeholderTextColor="#9ca3af"
              />
              <TouchableOpacity
                onPress={() => setShowNew(v => !v)}
                style={{ position: 'absolute', right: 12, top: 12 }}
              >
                <Text style={{ color: '#6b7280', fontSize: 13 }}>{showNew ? 'Hide' : 'Show'}</Text>
              </TouchableOpacity>
            </View>
            {newPassword.length > 0 && newPassword.length < 8 && (
              <Text style={{ color: '#dc2626', fontSize: 12, marginTop: 4 }}>Password must be at least 8 characters</Text>
            )}
          </View>

          <View style={styles.inputGroup}>
            <Text style={styles.label}>Confirm New Password</Text>
            <TextInput
              style={[styles.input, confirmPassword && newPassword !== confirmPassword ? { borderColor: '#dc2626' } : {}]}
              value={confirmPassword}
              onChangeText={setConfirmPassword}
              secureTextEntry
              placeholder="Repeat new password"
              placeholderTextColor="#9ca3af"
            />
            {confirmPassword.length > 0 && newPassword !== confirmPassword && (
              <Text style={{ color: '#dc2626', fontSize: 12, marginTop: 4 }}>Passwords do not match</Text>
            )}
          </View>

          <TouchableOpacity
            style={[styles.primaryButton, { marginTop: 8 }]}
            onPress={handleChangePassword}
            disabled={loading}
          >
            {loading
              ? <ActivityIndicator color="white" />
              : <Text style={styles.primaryButtonText}>🔒 Update Password</Text>
            }
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
};

const DetailedReportScreen = ({ navigateTo, selectedClass }) => {
  const [month, setMonth] = useState(new Date().getMonth() + 1);
  const [year, setYear] = useState(new Date().getFullYear());
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (selectedClass) {
      loadReport();
    }
  }, [month, year]);

  const loadReport = async () => {
    try {
      const result = await api.getClassReport(selectedClass.id, month, year);
      if (result.status === 'success') {
        setReport(result.data);
      }
    } catch (error) {
      Alert.alert('Error', 'Failed to load report');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadCSV = () => {
    Alert.alert('Download', 'CSV export functionality requires native file handling. Use web dashboard for CSV export.');
  };

  return (
    <View style={styles.screenContainer}>
      <View style={styles.header}>
        <TouchableOpacity
          onPress={() => navigateTo('TeacherDashboard')}
          style={{ padding: 10, marginRight: 8 }}
        >
          <ChevronLeft color="#374151" size={30} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Attendance Register</Text>
        <TouchableOpacity onPress={handleDownloadCSV}>
          <Download color="#374151" size={24} />
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scrollContent}>
        {/* Class Info */}
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>{selectedClass?.name}</Text>
          <Text style={styles.label}>Total Classes: {report?.total_classes || 0}</Text>
        </View>

        {/* Date Filters */}
        <View style={styles.filterRow}>
          <View style={styles.filterBox}>
            <Text>{new Date(year, month - 1).toLocaleString('default', { month: 'long' })}</Text>
            <ChevronDown size={16} color="#6b7280" />
          </View>
          <View style={[styles.filterBox, { width: '30%' }]}>
            <Text>{year}</Text>
            <ChevronDown size={16} color="#6b7280" />
          </View>
        </View>

        {/* Student List */}
        {loading ? (
          <ActivityIndicator size="large" color="#2563eb" style={{ marginTop: 20 }} />
        ) : (
          <View style={styles.card}>
            {report?.students?.map((student, index) => (
              <View key={index} style={styles.listItem}>
                <View>
                  <Text style={styles.listName}>{student.name}</Text>
                  <Text style={styles.listSub}>Roll: {student.roll_no}</Text>
                </View>
                <View style={{ alignItems: 'flex-end' }}>
                  <Text style={[
                    styles.listScore,
                    { color: student.attendance >= 75 ? '#16a34a' : '#dc2626' }
                  ]}>
                    {student.attendance}%
                  </Text>
                  <Text style={styles.editLink}>Edit</Text>
                </View>
              </View>
            ))}
          </View>
        )}
      </ScrollView>
    </View>
  );
};

// --- 6. NOTIFICATION HUB ---
const NotificationHubScreen = ({ navigateTo, selectedClass }) => {
  const [target, setTarget] = useState('defaulters');
  const [message, setMessage] = useState('Dear Student, your attendance is below the required threshold. Please meet the HOD immediately.');

  const handleSend = async () => {
    try {
      const result = await api.sendNotification(selectedClass?.id, target, message);
      if (result.status === 'success') {
        Alert.alert("Success", "Notifications sent successfully!");
      }
    } catch (error) {
      Alert.alert("Error", "Failed to send notifications");
    }
  };

  return (
    <View style={styles.screenContainer}>
      <View style={styles.header}>
        <TouchableOpacity
          onPress={() => navigateTo('TeacherDashboard')}
          style={{ padding: 10, marginRight: 8 }}
        >
          <ChevronLeft color="#374151" size={30} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Notification Hub</Text>
        <View style={{ width: 24 }} />
      </View>

      <ScrollView style={styles.scrollContent}>
        <View style={styles.card}>
          <Text style={styles.cardLabel}>TARGET AUDIENCE</Text>
          <View style={styles.grid2}>
            <TouchableOpacity
              style={[styles.gridItem, target === 'defaulters' && { backgroundColor: '#fef2f2', borderColor: '#fecaca' }]}
              onPress={() => setTarget('defaulters')}
            >
              <Text style={{ color: '#dc2626', fontWeight: 'bold' }}>⚠️ Defaulters</Text>
              <Text style={{ fontSize: 10, color: '#6b7280' }}>Attendance {'<'} 75%</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.gridItem, target === 'critical' && { backgroundColor: '#fef2f2', borderColor: '#fecaca' }]}
              onPress={() => setTarget('critical')}
            >
              <Text style={{ color: '#991b1b', fontWeight: 'bold' }}>🚨 Critical</Text>
              <Text style={{ fontSize: 10, color: '#6b7280' }}>Attendance {'<'} 50%</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.gridItem, target === 'all' && { backgroundColor: '#eff6ff', borderColor: '#bfdbfe' }]}
              onPress={() => setTarget('all')}
            >
              <Text style={{ color: '#2563eb', fontWeight: 'bold' }}>📢 All Class</Text>
              <Text style={{ fontSize: 10, color: '#6b7280' }}>General Notice</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.gridItem, target === 'select' && { backgroundColor: '#f3f4f6', borderColor: '#d1d5db' }]}
              onPress={() => setTarget('select')}
            >
              <Text style={{ color: '#374151', fontWeight: 'bold' }}>👤 Select</Text>
              <Text style={{ fontSize: 10, color: '#6b7280' }}>Individual</Text>
            </TouchableOpacity>
          </View>
        </View>

        <View style={styles.card}>
          <Text style={styles.cardLabel}>MESSAGE PREVIEW</Text>
          <TextInput
            style={styles.textArea}
            multiline
            numberOfLines={4}
            value={message}
            onChangeText={setMessage}
          />
        </View>

        <TouchableOpacity
          style={[styles.primaryButton, { backgroundColor: '#4f46e5' }]}
          onPress={handleSend}
        >
          <Text style={styles.primaryButtonText}>Send Notification</Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
};

// --- STYLES ---
const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f3f4f6' },
  contentContainer: { flex: 1, width: '100%', maxWidth: 480, alignSelf: 'center' },
  screenContainer: { flex: 1, backgroundColor: '#f9fafb' },
  scrollContent: { flex: 1, padding: 16 },

  // Header
  header: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', padding: 16, backgroundColor: 'white', borderBottomWidth: 1, borderBottomColor: '#e5e7eb' },
  headerTitle: { fontSize: 18, fontWeight: 'bold', color: '#111827' },
  headerSubtitle: { fontSize: 12, color: '#6b7280' },
  iconButton: { padding: 4 },
  notificationDot: { position: 'absolute', top: 4, right: 4, width: 8, height: 8, borderRadius: 4, backgroundColor: '#ef4444' },

  // Login
  loginContent: { flex: 1, justifyContent: 'center', padding: 24 },
  appTitle: { fontSize: 32, fontWeight: 'bold', color: '#1f2937', textAlign: 'center' },
  appSubtitle: { fontSize: 14, color: '#6b7280', textAlign: 'center', marginBottom: 48 },
  inputGroup: { marginBottom: 16 },
  label: { fontSize: 14, fontWeight: '500', color: '#374151', marginBottom: 6 },
  input: { backgroundColor: 'white', borderWidth: 1, borderColor: '#d1d5db', borderRadius: 8, padding: 12, fontSize: 16 },
  roleContainer: { marginBottom: 24 },
  roleToggle: { flexDirection: 'row', marginTop: 8, gap: 12 },
  roleButton: { flex: 1, padding: 12, borderWidth: 1, borderColor: '#d1d5db', borderRadius: 8, alignItems: 'center', backgroundColor: 'white' },
  roleButtonActive: { borderColor: '#2563eb', backgroundColor: '#eff6ff', borderWidth: 2 },
  roleText: { fontWeight: '600', color: '#374151' },
  roleTextActive: { color: '#2563eb' },
  primaryButton: { backgroundColor: '#2563eb', padding: 16, borderRadius: 12, alignItems: 'center', shadowOpacity: 0.1 },
  primaryButtonText: { color: 'white', fontWeight: 'bold', fontSize: 16 },
  helperText: { color: '#6b7280', fontSize: 12 },

  // Cards & Widgets
  card: { backgroundColor: 'white', borderRadius: 16, padding: 16, marginBottom: 16, borderWidth: 1, borderColor: '#e5e7eb' },
  cardLabel: { fontSize: 10, fontWeight: 'bold', color: '#9ca3af', marginBottom: 8, letterSpacing: 1 },

  // Session Picker
  pickerContainer: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', backgroundColor: '#f9fafb', padding: 12, borderRadius: 8, marginBottom: 12 },
  pickerText: { fontWeight: '600', color: '#374151', fontSize: 13 },
  statsRow: { flexDirection: 'row', gap: 8 },
  miniStat: { flex: 1, backgroundColor: '#f9fafb', padding: 8, borderRadius: 8, alignItems: 'center' },
  miniStatLabel: { fontSize: 8, color: '#6b7280', fontWeight: 'bold' },
  miniStatValue: { fontSize: 14, fontWeight: 'bold', color: '#1f2937' },

  // Action Grid
  actionGrid: { flexDirection: 'row', gap: 12, marginBottom: 16 },
  actionButton: { flex: 1, padding: 16, borderRadius: 16, alignItems: 'center', justifyContent: 'center', height: 100 },
  actionButtonText: { color: 'white', fontWeight: 'bold', textAlign: 'center', fontSize: 13 },

  // Reports
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 12 },
  sectionTitle: { fontSize: 16, fontWeight: 'bold', color: '#1f2937' },
  linkText: { color: '#2563eb', fontSize: 12, fontWeight: '600' },
  reportRow: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 },
  reportLabel: { fontSize: 13, fontWeight: '500', color: '#4b5563' },
  reportValue: { fontSize: 12, fontWeight: 'bold', paddingHorizontal: 8, paddingVertical: 2, borderRadius: 4, overflow: 'hidden' },
  outlineButton: { marginTop: 8, padding: 12, borderWidth: 1, borderColor: '#fecaca', borderRadius: 12, alignItems: 'center' },
  outlineButtonText: { color: '#dc2626', fontSize: 12, fontWeight: 'bold' },

  // Camera Screen
  liveBadge: { backgroundColor: 'rgba(234, 179, 8, 0.2)', paddingHorizontal: 8, paddingVertical: 4, borderRadius: 4, borderWidth: 1, borderColor: 'rgba(234,179,8,0.3)' },
  liveText: { color: '#facc15', fontSize: 10, fontWeight: 'bold' },
  cameraContainer: { flex: 1, margin: 16, borderRadius: 24, overflow: 'hidden', borderWidth: 2, borderColor: '#374151' },
  camera: { flex: 1 },
  preview: { flex: 1, resizeMode: 'contain' },
  overlay: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  overlayText: { color: 'rgba(255,255,255,0.5)', fontSize: 20, fontWeight: 'bold', borderStyle: 'dashed', borderWidth: 2, borderColor: 'rgba(255,255,255,0.5)', padding: 20, borderRadius: 12 },
  flipBtn: { position: 'absolute', bottom: 20, right: 20, backgroundColor: 'rgba(0,0,0,0.6)', padding: 8, borderRadius: 8 },
  resultCard: { padding: 16, borderRadius: 16, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 },
  bgGreen: { backgroundColor: '#22c55e' },
  bgRed: { backgroundColor: '#ef4444' },
  resultText: { color: 'white', fontWeight: 'bold', fontSize: 16 },
  resultBadge: { backgroundColor: 'rgba(255,255,255,0.2)', padding: 6, borderRadius: 8 },

  // Student Dash
  profileCard: { backgroundColor: '#4f46e5', borderRadius: 20, padding: 20, flexDirection: 'row', alignItems: 'center', marginBottom: 16 },
  avatar: { width: 50, height: 50, borderRadius: 25, backgroundColor: 'rgba(255,255,255,0.2)', alignItems: 'center', justifyContent: 'center', borderTopWidth: 1, borderColor: 'rgba(255,255,255,0.4)' },
  profileName: { color: 'white', fontSize: 18, fontWeight: 'bold', marginLeft: 12 },
  profileDetail: { color: 'rgba(255,255,255,0.8)', fontSize: 12, marginLeft: 12 },
  profileValueText: { fontSize: 15, color: '#111827', paddingVertical: 8, paddingHorizontal: 2 },
  statValue: { fontSize: 24, fontWeight: 'bold' },
  progressBarBg: { height: 8, backgroundColor: '#f3f4f6', borderRadius: 4, overflow: 'hidden', marginBottom: 8 },
  progressBarFill: { height: '100%', borderRadius: 4 },
  warningBadge: { backgroundColor: '#fef2f2', padding: 8, borderRadius: 8, alignItems: 'center' },
  warningText: { color: '#dc2626', fontSize: 11, fontWeight: '600' },
  subjectCard: { backgroundColor: 'white', padding: 16, borderRadius: 12, marginBottom: 8, flexDirection: 'row', justifyContent: 'space-between', borderWidth: 1, borderColor: '#f3f4f6' },
  subjectTitle: { fontSize: 14, fontWeight: 'bold', color: '#1f2937' },
  subjectDetail: { fontSize: 10, color: '#9ca3af' },
  subjectValue: { fontSize: 16, fontWeight: 'bold' },
  statusTag: { fontSize: 10, paddingHorizontal: 6, paddingVertical: 2, borderRadius: 4, overflow: 'hidden', marginTop: 2, fontWeight: 'bold' },

  // Report Screen
  filterRow: { flexDirection: 'row', gap: 8, marginBottom: 16 },
  filterBox: { flex: 1, backgroundColor: 'white', padding: 10, borderRadius: 8, borderWidth: 1, borderColor: '#e5e7eb', flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  listItem: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 12, borderBottomWidth: 1, borderBottomColor: '#f3f4f6' },
  listName: { fontWeight: '600', color: '#1f2937', fontSize: 14 },
  listSub: { fontSize: 12, color: '#9ca3af' },
  listScore: { fontWeight: 'bold', fontSize: 14 },
  editLink: { fontSize: 10, color: '#2563eb', fontWeight: '500', textDecorationLine: 'underline', marginTop: 2 },

  // Notification Hub
  grid2: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  gridItem: { width: '48%', padding: 12, borderRadius: 12, borderWidth: 1, borderColor: '#e5e7eb' },
  textArea: { backgroundColor: '#f9fafb', borderWidth: 1, borderColor: '#e5e7eb', borderRadius: 8, padding: 12, fontSize: 14, color: '#4b5563', textAlignVertical: 'top' },

  // Bottom Nav
  bottomNav: { flexDirection: 'row', justifyContent: 'space-around', padding: 12, backgroundColor: 'white', borderTopWidth: 1, borderTopColor: '#e5e7eb' },
  navItem: { alignItems: 'center' },
  navText: { fontSize: 10, marginTop: 4, color: '#9ca3af', fontWeight: '500' },

  // Modal Styles
  modalOverlay: { flex: 1, backgroundColor: 'rgba(0,0,0,0.5)', justifyContent: 'flex-end' },
  modalContent: { backgroundColor: 'white', borderTopLeftRadius: 24, borderTopRightRadius: 24, padding: 24, maxHeight: '70%' },
  modalTitle: { fontSize: 20, fontWeight: 'bold', color: '#1f2937', marginBottom: 16 },
  modalItem: { padding: 16, borderBottomWidth: 1, borderBottomColor: '#f3f4f6' },
  modalItemSelected: { backgroundColor: '#eff6ff', borderLeftWidth: 4, borderLeftColor: '#2563eb' },
  modalItemText: { fontSize: 16, fontWeight: '600', color: '#1f2937' },
  modalItemSubtext: { fontSize: 12, color: '#6b7280', marginTop: 4 },
  modalCloseButton: { marginTop: 16, padding: 16, backgroundColor: '#f3f4f6', borderRadius: 12, alignItems: 'center' },
  modalCloseButtonText: { fontSize: 14, fontWeight: '600', color: '#6b7280' },
});