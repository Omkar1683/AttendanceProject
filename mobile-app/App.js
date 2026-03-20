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
  const [email, setEmail] = useState('prof.XYZ@ves.ac.in');
  const [password, setPassword] = useState('teacher123');
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

  // Add Subject modal state
  const [showAddSubject, setShowAddSubject] = useState(false);
  const [newSubjectName, setNewSubjectName] = useState('');
  const [newSubjectCode, setNewSubjectCode] = useState('');
  const [newSubjectStudents, setNewSubjectStudents] = useState('');
  const [newSubjectBatch, setNewSubjectBatch] = useState('');
  const [newSubjectDept, setNewSubjectDept] = useState('');
  const [addingSubject, setAddingSubject] = useState(false);

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

  const handleAddSubject = async () => {
    if (!newSubjectName.trim() || !newSubjectCode.trim()) {
      Alert.alert('Validation Error', 'Subject Name and Code are required.');
      return;
    }
    const totalStudents = parseInt(newSubjectStudents, 10);
    if (isNaN(totalStudents) || totalStudents < 1) {
      Alert.alert('Validation Error', 'Enter a valid number of students (≥ 1).');
      return;
    }
    setAddingSubject(true);
    try {
      const result = await api.createClass({
        name: newSubjectName.trim(),
        code: newSubjectCode.trim().toUpperCase(),
        total_students: totalStudents,
        batch: newSubjectBatch.trim() || undefined,
        department: newSubjectDept.trim() || undefined,
      });
      if (result.status === 'success') {
        Alert.alert('Success', `Subject "${newSubjectName}" created!`);
        // Reset form
        setNewSubjectName(''); setNewSubjectCode('');
        setNewSubjectStudents(''); setNewSubjectBatch(''); setNewSubjectDept('');
        setShowAddSubject(false);
        // Reload class dropdown
        const classesResult = await api.getClasses(userInfo.id);
        if (classesResult.status === 'success') {
          setClasses(classesResult.data);
          // Auto-select newly created class
          const newClass = classesResult.data.find(c => c.id === result.class_id);
          if (newClass) setSelectedClassId(newClass.id);
        }
      } else {
        Alert.alert('Error', result.message || 'Could not create subject');
      }
    } catch (err) {
      Alert.alert('Error', err.message || 'Network error');
    } finally {
      setAddingSubject(false);
    }
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
      {/* ── Class Picker Modal ──────────────────────────────────────── */}
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
                    <Text style={styles.modalItemText}>{cls.name} ({cls.code})</Text>
                    <Text style={styles.modalItemSubtext}>{cls.batch} • {cls.total_students} students</Text>
                  </TouchableOpacity>
                ))
              ) : (
                <View style={{ padding: 32, alignItems: 'center' }}>
                  <Text style={{ color: '#6b7280', fontSize: 14, textAlign: 'center' }}>No classes found.{'\n'}Tap "+ Add Subject" below.</Text>
                </View>
              )}
            </ScrollView>
            <TouchableOpacity
              style={[styles.modalCloseButton, { backgroundColor: '#eff6ff', marginBottom: 8 }]}
              onPress={() => { setShowClassPicker(false); setShowAddSubject(true); }}
            >
              <Text style={{ color: '#2563eb', fontWeight: 'bold', fontSize: 14 }}>+ Add New Subject</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.modalCloseButton}
              onPress={() => setShowClassPicker(false)}
            >
              <Text style={styles.modalCloseButtonText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* ── Add Subject Modal ──────────────────────────────────────── */}
      <Modal
        visible={showAddSubject}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setShowAddSubject(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { maxHeight: '90%' }]}>
            <Text style={styles.modalTitle}>➕ Add New Subject</Text>
            <ScrollView style={{ maxHeight: 420 }} showsVerticalScrollIndicator={false}>
              <View style={styles.inputGroup}>
                <Text style={styles.label}>Subject Name *</Text>
                <TextInput
                  style={styles.input}
                  placeholder="e.g., Data Structures"
                  placeholderTextColor="#9ca3af"
                  value={newSubjectName}
                  onChangeText={setNewSubjectName}
                />
              </View>
              <View style={styles.inputGroup}>
                <Text style={styles.label}>Subject Code *</Text>
                <TextInput
                  style={styles.input}
                  placeholder="e.g., CS301"
                  placeholderTextColor="#9ca3af"
                  value={newSubjectCode}
                  onChangeText={setNewSubjectCode}
                  autoCapitalize="characters"
                />
              </View>
              <View style={styles.inputGroup}>
                <Text style={styles.label}>Total Students *</Text>
                <TextInput
                  style={styles.input}
                  placeholder="e.g., 60"
                  placeholderTextColor="#9ca3af"
                  value={newSubjectStudents}
                  onChangeText={setNewSubjectStudents}
                  keyboardType="number-pad"
                />
              </View>
              <View style={styles.inputGroup}>
                <Text style={styles.label}>Batch</Text>
                <TextInput
                  style={styles.input}
                  placeholder="e.g., MCA 2A"
                  placeholderTextColor="#9ca3af"
                  value={newSubjectBatch}
                  onChangeText={setNewSubjectBatch}
                />
              </View>
              <View style={styles.inputGroup}>
                <Text style={styles.label}>Department</Text>
                <TextInput
                  style={styles.input}
                  placeholder="e.g., MCA"
                  placeholderTextColor="#9ca3af"
                  value={newSubjectDept}
                  onChangeText={setNewSubjectDept}
                />
              </View>
            </ScrollView>
            <TouchableOpacity
              style={[styles.primaryButton, { marginTop: 12 }]}
              onPress={handleAddSubject}
              disabled={addingSubject}
            >
              {addingSubject ? (
                <ActivityIndicator color="white" />
              ) : (
                <Text style={styles.primaryButtonText}>✓ Create Subject</Text>
              )}
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.modalCloseButton, { marginTop: 8 }]}
              onPress={() => setShowAddSubject(false)}
            >
              <Text style={styles.modalCloseButtonText}>Cancel</Text>
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
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <Text style={styles.cardLabel}>SELECT SESSION</Text>
            <TouchableOpacity onPress={() => setShowAddSubject(true)}>
              <Text style={{ color: '#2563eb', fontSize: 12, fontWeight: '700' }}>+ Add Subject</Text>
            </TouchableOpacity>
          </View>
          <TouchableOpacity
            style={styles.pickerContainer}
            onPress={() => setShowClassPicker(true)}
          >
            <Text style={styles.pickerText}>
              {selectedClassData ? `${selectedClassData.name} (${selectedClassData.code})` : 'Select a class'}
            </Text>
            <ChevronDown color="#4b5563" size={20} />
          </TouchableOpacity>

          {selectedClassData && (
            <View style={styles.statsRow}>
              <View style={styles.miniStat}>
                <Text style={styles.miniStatLabel}>BATCH</Text>
                <Text style={styles.miniStatValue}>{selectedClassData.batch || '—'}</Text>
              </View>
              <View style={styles.miniStat}>
                <Text style={styles.miniStatLabel}>CODE</Text>
                <Text style={styles.miniStatValue}>{selectedClassData.code}</Text>
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

        {/* Action Buttons — 2x2 Grid */}
        <View style={{ flexDirection: 'row', flexWrap: 'wrap', gap: 12, marginBottom: 16 }}>
          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: '#16a34a', width: '47%' }]}
            onPress={handleStartAttendance}
          >
            <Camera color="white" size={28} style={{ marginBottom: 6 }} />
            <Text style={styles.actionButtonText}>Start{'\n'}Attendance</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: '#6366f1', width: '47%' }]}
            onPress={() => {
              setSelectedClass(selectedClassData);
              navigateTo('DetailedReport');
            }}
          >
            <BarChart3 color="white" size={28} style={{ marginBottom: 6 }} />
            <Text style={styles.actionButtonText}>View{'\n'}Reports</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: '#ea580c', width: '47%' }]}
            onPress={() => navigateTo('RegisterStudent')}
          >
            <Camera color="white" size={28} style={{ marginBottom: 6 }} />
            <Text style={styles.actionButtonText}>Register{'\n'}Student</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: '#0891b2', width: '47%' }]}
            onPress={() => setShowAddSubject(true)}
          >
            <Text style={{ fontSize: 26, marginBottom: 6 }}>📚</Text>
            <Text style={styles.actionButtonText}>Add{'\n'}Subject</Text>
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

// --- 4. STUDENT DASHBOARD ---
const StudentDashboard = ({ navigateTo, userInfo, onLogout }) => {
  const [studentReport, setStudentReport] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStudentData();
  }, []);

  const loadStudentData = async () => {
    try {
      // In a real app, you'd map the user ID to student ID
      // For now, using a placeholder
      const result = await api.getStudentReport(userInfo.id);
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
      <View style={styles.header}>
        <TouchableOpacity onPress={onLogout} style={{ padding: 10 }}>
          <LogOut color="#6b7280" size={28} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>My Attendance</Text>
        <Bell color="#6b7280" size={24} />
      </View>

      <ScrollView style={styles.scrollContent}>
        {/* Profile */}
        <View style={styles.profileCard}>
          <View style={styles.avatar}>
            <Text style={{ color: 'white', fontSize: 20 }}>👤</Text>
          </View>
          <View>
            <Text style={styles.profileName}>{userInfo?.name}</Text>
            <Text style={styles.profileDetail}>Roll No: {userInfo?.roll_no || 'N/A'} | {userInfo?.department}</Text>
          </View>
        </View>

        {/* Stats */}
        <View style={styles.card}>
          <Text style={styles.cardLabel}>OVERALL PERFORMANCE</Text>
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: 8 }}>
            <Text style={styles.label}>Current Percentage</Text>
            <Text style={[styles.statValue, { color: isDefaulter ? '#dc2626' : '#16a34a' }]}>
              {overallPercentage}%
            </Text>
          </View>
          <View style={styles.progressBarBg}>
            <View style={[
              styles.progressBarFill,
              { width: `${overallPercentage}%`, backgroundColor: isDefaulter ? '#dc2626' : '#16a34a' }
            ]} />
          </View>
          {isDefaulter && (
            <View style={styles.warningBadge}>
              <Text style={styles.warningText}>⚠ Below required minimum (75%)</Text>
            </View>
          )}
        </View>

        {/* Subject Breakdown */}
        <Text style={styles.sectionTitle}>Subject Breakdown</Text>
        {studentReport?.subjects?.map((subject, index) => (
          <View key={index} style={styles.subjectCard}>
            <View>
              <Text style={styles.subjectTitle}>{subject.name}</Text>
              <Text style={styles.subjectDetail}>Total Classes: {subject.total_classes}</Text>
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

// --- 5. DETAILED REPORT SCREEN ---
const MONTHS = [
  { label: 'January', value: 1 }, { label: 'February', value: 2 },
  { label: 'March', value: 3 },   { label: 'April', value: 4 },
  { label: 'May', value: 5 },     { label: 'June', value: 6 },
  { label: 'July', value: 7 },    { label: 'August', value: 8 },
  { label: 'September', value: 9 },{ label: 'October', value: 10 },
  { label: 'November', value: 11 },{ label: 'December', value: 12 },
];
const YEARS = Array.from({ length: 5 }, (_, i) => new Date().getFullYear() - i);

const DetailedReportScreen = ({ navigateTo, selectedClass }) => {
  const [month, setMonth] = useState(new Date().getMonth() + 1);
  const [year, setYear] = useState(new Date().getFullYear());
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [csvLoading, setCsvLoading] = useState(false);

  // Month / Year picker modal state
  const [showMonthPicker, setShowMonthPicker] = useState(false);
  const [showYearPicker, setShowYearPicker] = useState(false);

  // Edit attendance modal state
  const [editStudent, setEditStudent] = useState(null);        // { student_id, name, session_id, currentStatus }
  const [editStatus, setEditStatus] = useState('Present');
  const [editLoading, setEditLoading] = useState(false);

  // Fetch report whenever class / month / year changes
  useEffect(() => {
    if (selectedClass?.id) {
      loadReport(selectedClass.id, month, year);
    }
  }, [selectedClass, month, year]);

  const loadReport = async (classId, m, y) => {
    setLoading(true);
    try {
      const result = await api.getClassReport(classId, m, y);
      if (result.status === 'success') {
        setReport(result.data);
      } else {
        Alert.alert('Error', result.message || 'Failed to load report');
      }
    } catch (error) {
      Alert.alert('Error', 'Network error loading report');
    } finally {
      setLoading(false);
    }
  };

  const handleMonthSelect = (m) => {
    setMonth(m);
    setShowMonthPicker(false);
  };

  const handleYearSelect = (y) => {
    setYear(y);
    setShowYearPicker(false);
  };

  // Open Edit modal for a student
  const openEditModal = (student) => {
    // We need a session_id for this month — use the first session in the report
    // The backend manual endpoint uses session_id to identify which record to update
    // We'll pass a "virtual" session reference: the report includes sessions for this month
    // For manual edit we use the most recent session from report.sessions if available
    const sessionId = report?.latest_session_id || report?.sessions?.[0] || null;
    setEditStudent({
      student_id: student.student_id,
      name: student.name,
      session_id: sessionId,
      currentStatus: student.status,  // 'Good' or 'Defaulter' — attendance %
      attendance: student.attendance,
    });
    setEditStatus(student.attendance > 0 ? 'Present' : 'Absent');
  };

  const handleSaveEdit = async () => {
    if (!editStudent?.session_id) {
      Alert.alert(
        'No Session Found',
        'Cannot edit attendance for a month with no recorded sessions. Start a session first.'
      );
      return;
    }
    setEditLoading(true);
    try {
      const result = await api.manualAttendance(
        editStudent.student_id,
        editStudent.session_id,
        editStatus
      );
      if (result.status === 'success') {
        // Instant UI update — update the student's status in local state
        setReport(prev => {
          if (!prev) return prev;
          const updatedStudents = prev.students.map(s => {
            if (s.student_id === editStudent.student_id) {
              const newPresent = editStatus === 'Present'
                ? Math.min(s.present + 1, s.total)
                : Math.max(s.present - 1, 0);
              const newPct = prev.total_classes > 0
                ? Math.round((newPresent / prev.total_classes) * 100 * 100) / 100
                : 0;
              return {
                ...s,
                present: newPresent,
                absent: s.total - newPresent,
                attendance: newPct,
                status: newPct >= 75 ? 'Good' : 'Defaulter',
              };
            }
            return s;
          });
          return { ...prev, students: updatedStudents };
        });
        Alert.alert('Saved', `${editStudent.name} marked as ${editStatus}`);
        setEditStudent(null);
      } else {
        Alert.alert('Error', result.message || 'Could not save');
      }
    } catch (err) {
      Alert.alert('Error', err.message || 'Network error');
    } finally {
      setEditLoading(false);
    }
  };

  const handleDownloadCSV = async () => {
    if (!selectedClass?.id) {
      Alert.alert('Error', 'No class selected');
      return;
    }
    setCsvLoading(true);
    try {
      await api.downloadAndShareCSV(
        selectedClass.id,
        month,
        year,
        selectedClass.name
      );
    } catch (err) {
      Alert.alert('Download Failed', err.message || 'Could not download CSV');
    } finally {
      setCsvLoading(false);
    }
  };

  const selectedMonthLabel = MONTHS.find(m => m.value === month)?.label || '';

  return (
    <View style={styles.screenContainer}>

      {/* ── Month Picker Modal ─────────────────────────── */}
      <Modal visible={showMonthPicker} transparent animationType="slide"
        onRequestClose={() => setShowMonthPicker(false)}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Select Month</Text>
            <ScrollView style={{ maxHeight: 380 }}>
              {MONTHS.map(m => (
                <TouchableOpacity
                  key={m.value}
                  style={[styles.modalItem, m.value === month && styles.modalItemSelected]}
                  onPress={() => handleMonthSelect(m.value)}
                >
                  <Text style={[styles.modalItemText, m.value === month && { color: '#2563eb' }]}>
                    {m.label}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
            <TouchableOpacity style={styles.modalCloseButton} onPress={() => setShowMonthPicker(false)}>
              <Text style={styles.modalCloseButtonText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* ── Year Picker Modal ──────────────────────────── */}
      <Modal visible={showYearPicker} transparent animationType="slide"
        onRequestClose={() => setShowYearPicker(false)}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Select Year</Text>
            <ScrollView style={{ maxHeight: 280 }}>
              {YEARS.map(y => (
                <TouchableOpacity
                  key={y}
                  style={[styles.modalItem, y === year && styles.modalItemSelected]}
                  onPress={() => handleYearSelect(y)}
                >
                  <Text style={[styles.modalItemText, y === year && { color: '#2563eb' }]}>{y}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
            <TouchableOpacity style={styles.modalCloseButton} onPress={() => setShowYearPicker(false)}>
              <Text style={styles.modalCloseButtonText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* ── Edit Attendance Modal ──────────────────────── */}
      <Modal visible={!!editStudent} transparent animationType="fade"
        onRequestClose={() => setEditStudent(null)}>
        <View style={[styles.modalOverlay, { justifyContent: 'center', paddingHorizontal: 24 }]}>
          <View style={[styles.modalContent, { borderRadius: 20 }]}>
            <Text style={styles.modalTitle}>Edit Attendance</Text>
            <Text style={{ color: '#6b7280', fontSize: 14, marginBottom: 16 }}>
              {editStudent?.name}  •  Current: {editStudent?.attendance}%
            </Text>

            {/* Toggle Present / Absent */}
            <View style={{ flexDirection: 'row', gap: 12, marginBottom: 20 }}>
              <TouchableOpacity
                style={[{
                  flex: 1, padding: 16, borderRadius: 12, alignItems: 'center',
                  borderWidth: 2,
                  borderColor: editStatus === 'Present' ? '#16a34a' : '#e5e7eb',
                  backgroundColor: editStatus === 'Present' ? '#dcfce7' : '#f9fafb',
                }]}
                onPress={() => setEditStatus('Present')}
              >
                <Text style={{ fontSize: 22, marginBottom: 4 }}>✅</Text>
                <Text style={{
                  fontWeight: 'bold', fontSize: 14,
                  color: editStatus === 'Present' ? '#16a34a' : '#9ca3af'
                }}>
                  Present
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={[{
                  flex: 1, padding: 16, borderRadius: 12, alignItems: 'center',
                  borderWidth: 2,
                  borderColor: editStatus === 'Absent' ? '#dc2626' : '#e5e7eb',
                  backgroundColor: editStatus === 'Absent' ? '#fee2e2' : '#f9fafb',
                }]}
                onPress={() => setEditStatus('Absent')}
              >
                <Text style={{ fontSize: 22, marginBottom: 4 }}>❌</Text>
                <Text style={{
                  fontWeight: 'bold', fontSize: 14,
                  color: editStatus === 'Absent' ? '#dc2626' : '#9ca3af'
                }}>
                  Absent
                </Text>
              </TouchableOpacity>
            </View>

            <TouchableOpacity
              style={[styles.primaryButton, { marginBottom: 8 }]}
              onPress={handleSaveEdit}
              disabled={editLoading}
            >
              {editLoading
                ? <ActivityIndicator color="white" />
                : <Text style={styles.primaryButtonText}>💾 Save Change</Text>
              }
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.modalCloseButton}
              onPress={() => setEditStudent(null)}
            >
              <Text style={styles.modalCloseButtonText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigateTo('TeacherDashboard')}
          style={{ padding: 10, marginRight: 8 }}>
          <ChevronLeft color="#374151" size={30} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Attendance Register</Text>
        <TouchableOpacity onPress={handleDownloadCSV} disabled={csvLoading}>
          {csvLoading
            ? <ActivityIndicator size="small" color="#374151" />
            : <Download color="#374151" size={24} />
          }
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scrollContent}>
        {/* Class Info */}
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>{selectedClass?.name}</Text>
          <Text style={styles.label}>Total Sessions: {report?.total_classes || 0}</Text>
        </View>

        {/* ── Filter Row — Month & Year (INTERACTIVE) ── */}
        <View style={styles.filterRow}>
          <TouchableOpacity
            style={[styles.filterBox, { flex: 1 }]}
            onPress={() => setShowMonthPicker(true)}
          >
            <Text style={{ fontSize: 14, color: '#111827', fontWeight: '500' }}>
              {selectedMonthLabel}
            </Text>
            <ChevronDown size={16} color="#6b7280" />
          </TouchableOpacity>

          <TouchableOpacity
            style={[styles.filterBox, { width: '32%' }]}
            onPress={() => setShowYearPicker(true)}
          >
            <Text style={{ fontSize: 14, color: '#111827', fontWeight: '500' }}>{year}</Text>
            <ChevronDown size={16} color="#6b7280" />
          </TouchableOpacity>
        </View>

        {/* CSV Download Banner */}
        <TouchableOpacity
          style={{
            flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
            backgroundColor: '#eff6ff', padding: 10, borderRadius: 10, marginBottom: 12,
            borderWidth: 1, borderColor: '#bfdbfe',
          }}
          onPress={handleDownloadCSV}
          disabled={csvLoading}
        >
          <Download color="#2563eb" size={16} style={{ marginRight: 8 }} />
          <Text style={{ color: '#2563eb', fontWeight: '600', fontSize: 13 }}>
            {csvLoading ? 'Preparing CSV...' : `Download CSV — ${selectedMonthLabel} ${year}`}
          </Text>
        </TouchableOpacity>

        {/* Student List */}
        {loading ? (
          <ActivityIndicator size="large" color="#2563eb" style={{ marginTop: 40 }} />
        ) : !report?.students?.length ? (
          <View style={[styles.card, { alignItems: 'center', padding: 32 }]}>
            <Text style={{ color: '#6b7280', fontSize: 14 }}>
              No sessions found for {selectedMonthLabel} {year}
            </Text>
          </View>
        ) : (
          <View style={styles.card}>
            <View style={[styles.cardHeader, { marginBottom: 12 }]}>
              <Text style={styles.sectionTitle}>
                Students ({report.students.length})
              </Text>
              <Text style={styles.label}>{report.total_classes} sessions</Text>
            </View>
            {report.students.map((student, index) => (
              <View key={index} style={[styles.listItem, {
                backgroundColor: index % 2 === 0 ? '#ffffff' : '#f9fafb',
                paddingHorizontal: 8, borderRadius: 8,
              }]}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.listName}>{student.name}</Text>
                  <Text style={styles.listSub}>
                    Roll: {student.roll_no}  •  {student.present}/{student.total} present
                  </Text>
                </View>
                <View style={{ alignItems: 'flex-end' }}>
                  <Text style={[
                    styles.listScore,
                    { color: student.attendance >= 75 ? '#16a34a' : '#dc2626' }
                  ]}>
                    {student.attendance}%
                  </Text>
                  <TouchableOpacity
                    style={{
                      marginTop: 4, paddingHorizontal: 10, paddingVertical: 3,
                      borderRadius: 6, borderWidth: 1,
                      borderColor: '#2563eb', backgroundColor: '#eff6ff'
                    }}
                    onPress={() => openEditModal(student)}
                  >
                    <Text style={{ color: '#2563eb', fontSize: 11, fontWeight: '700' }}>✏ Edit</Text>
                  </TouchableOpacity>
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