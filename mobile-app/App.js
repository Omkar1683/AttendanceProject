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
import { Ionicons } from '@expo/vector-icons';

// Icon shim: maps lucide icon components to Ionicons equivalents
const Camera     = ({ size, color, style }) => <Ionicons name="camera-outline"         size={size} color={color} style={style} />;
const BarChart3  = ({ size, color, style }) => <Ionicons name="bar-chart-outline"      size={size} color={color} style={style} />;
const Home       = ({ size, color, style }) => <Ionicons name="home-outline"           size={size} color={color} style={style} />;
const Settings   = ({ size, color, style }) => <Ionicons name="settings-outline"       size={size} color={color} style={style} />;
const ChevronLeft= ({ size, color, style }) => <Ionicons name="chevron-back-outline"   size={size} color={color} style={style} />;
const Bell       = ({ size, color, style }) => <Ionicons name="notifications-outline"  size={size} color={color} style={style} />;
const Calendar   = ({ size, color, style }) => <Ionicons name="calendar-outline"       size={size} color={color} style={style} />;
const LogOut     = ({ size, color, style }) => <Ionicons name="log-out-outline"        size={size} color={color} style={style} />;
const ChevronDown= ({ size, color, style }) => <Ionicons name="chevron-down-outline"   size={size} color={color} style={style} />;
const Download   = ({ size, color, style }) => <Ionicons name="download-outline"       size={size} color={color} style={style} />;
import { api } from './utils/api';
import StudentApp from './student-module/StudentApp';

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
        return <StudentApp userInfo={userInfo} onLogout={handleLogout} />;
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

// --- 2. SIGNUP SCREEN (Teacher accounts only) ---
const SignupScreen = ({ navigateTo }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [department, setDepartment] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSignup = async () => {
    if (!email || !password || !name) {
      Alert.alert('Error', 'Name, email and password are required');
      return;
    }
    if (password.length < 8) {
      Alert.alert('Error', 'Password must be at least 8 characters');
      return;
    }

    setLoading(true);
    try {
      // Role is always 'teacher' — students are created by teachers via Register Student
      await api.signup(email, password, name, department);
      Alert.alert('Success', 'Teacher account created successfully!', [
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
        <Text style={styles.appSubtitle}>Teacher Registration — AttendAI</Text>

        <View style={[styles.card, { backgroundColor: '#eff6ff', borderColor: '#bfdbfe', marginBottom: 20 }]}>
          <Text style={{ color: '#1d4ed8', fontSize: 13, fontWeight: '600', textAlign: 'center' }}>
            🎓 This page is for Teacher accounts only.{`\n`}Students are added by teachers via "Register Student".
          </Text>
        </View>

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
          <Text style={styles.label}>Password * (min 8 characters)</Text>
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
          <Text style={styles.label}>Department</Text>
          <TextInput
            style={styles.input}
            placeholder="e.g., MCA, Computer Science"
            placeholderTextColor="#9ca3af"
            value={department}
            onChangeText={setDepartment}
          />
        </View>

        <TouchableOpacity
          style={styles.primaryButton}
          onPress={handleSignup}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="white" />
          ) : (
            <Text style={styles.primaryButtonText}>Create Teacher Account</Text>
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
  const [newSubjectBatch, setNewSubjectBatch] = useState('');
  const [newSubjectDept, setNewSubjectDept] = useState('');
  const [addingSubject, setAddingSubject] = useState(false);
  // Student multi-select state (shared by Add Subject & Update Class)
  const [availableStudents, setAvailableStudents] = useState([]);
  const [selectedStudentIds, setSelectedStudentIds] = useState([]);
  const [studentsLoading, setStudentsLoading] = useState(false);

  // Update Class modal state
  const [showUpdateClass, setShowUpdateClass] = useState(false);
  const [updateClassBatch, setUpdateClassBatch] = useState('');
  const [updatingClass, setUpdatingClass] = useState(false);

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

  const loadStudentsForPicker = async () => {
    setStudentsLoading(true);
    try {
      const result = await api.getStudents();
      if (result.status === 'success') {
        setAvailableStudents(result.data || []);
      }
    } catch (err) {
      console.warn('Could not load students:', err.message);
    } finally {
      setStudentsLoading(false);
    }
  };

  const toggleStudentSelection = (studentId) => {
    setSelectedStudentIds(prev =>
      prev.includes(studentId)
        ? prev.filter(id => id !== studentId)
        : [...prev, studentId]
    );
  };

  const handleAddSubject = async () => {
    if (!newSubjectName.trim() || !newSubjectCode.trim()) {
      Alert.alert('Validation Error', 'Subject Name and Code are required.');
      return;
    }
    setAddingSubject(true);
    try {
      const result = await api.createClass({
        name: newSubjectName.trim(),
        code: newSubjectCode.trim().toUpperCase(),
        student_ids: selectedStudentIds,
        batch: newSubjectBatch.trim() || undefined,
        department: newSubjectDept.trim() || undefined,
      });
      if (result.status === 'success') {
        Alert.alert('Success', `Subject "${newSubjectName}" created with ${selectedStudentIds.length} student(s)!`);
        // Reset form
        setNewSubjectName(''); setNewSubjectCode('');
        setNewSubjectBatch(''); setNewSubjectDept('');
        setSelectedStudentIds([]);
        setShowAddSubject(false);
        // Reload class dropdown
        const classesResult = await api.getClasses(userInfo.id);
        if (classesResult.status === 'success') {
          setClasses(classesResult.data);
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

  // ── Update Class handlers ─────────────────────────────────────────────────
  const handleOpenUpdateClass = async () => {
    if (!selectedClassData) {
      Alert.alert('Error', 'Please select a class first');
      return;
    }
    setUpdateClassBatch(selectedClassData.batch || '');
    // Pre-select currently enrolled students
    setSelectedStudentIds(selectedClassData.students || []);
    setShowUpdateClass(true);
    await loadStudentsForPicker();
  };

  const handleSaveUpdateClass = async () => {
    if (!selectedClassData) return;
    setUpdatingClass(true);
    try {
      const result = await api.updateClass(selectedClassData.id, {
        students: selectedStudentIds,
        batch: updateClassBatch.trim() || undefined,
      });
      if (result.status === 'success') {
        Alert.alert('Success', `Class updated with ${selectedStudentIds.length} student(s)!`);
        setShowUpdateClass(false);
        // Refresh class data
        const classesResult = await api.getClasses(userInfo.id);
        if (classesResult.status === 'success') {
          setClasses(classesResult.data);
          // Refresh summary for updated class
          if (selectedClassId) loadClassData(selectedClassId);
        }
      } else {
        Alert.alert('Error', result.message || 'Could not update class');
      }
    } catch (err) {
      Alert.alert('Error', err.message || 'Network error');
    } finally {
      setUpdatingClass(false);
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
        onShow={loadStudentsForPicker}
      >
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { maxHeight: '92%' }]}>
            <Text style={styles.modalTitle}>➕ Add New Subject</Text>
            <ScrollView style={{ maxHeight: 480 }} showsVerticalScrollIndicator={false}>
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

              {/* ── Student Multi-Select ─────────────────────────────── */}
              <View style={styles.inputGroup}>
                <View style={{ flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <Text style={styles.label}>Assign Students</Text>
                  <Text style={{ fontSize: 12, color: '#2563eb', fontWeight: '600' }}>
                    {selectedStudentIds.length} selected
                  </Text>
                </View>
                {studentsLoading ? (
                  <ActivityIndicator color="#2563eb" />
                ) : availableStudents.length === 0 ? (
                  <Text style={{ color: '#9ca3af', fontSize: 13, fontStyle: 'italic', textAlign: 'center', padding: 12 }}>
                    No students registered yet.
                  </Text>
                ) : (
                  <View style={{ borderWidth: 1, borderColor: '#e5e7eb', borderRadius: 8, overflow: 'hidden', maxHeight: 200 }}>
                    <ScrollView nestedScrollEnabled={true}>
                      {availableStudents.map((student) => {
                        const isSelected = selectedStudentIds.includes(student.id);
                        return (
                          <TouchableOpacity
                            key={student.id}
                            style={[{
                              flexDirection: 'row', alignItems: 'center',
                              padding: 10, borderBottomWidth: 1, borderBottomColor: '#f3f4f6',
                              backgroundColor: isSelected ? '#eff6ff' : '#fff',
                            }]}
                            onPress={() => toggleStudentSelection(student.id)}
                          >
                            <View style={[{
                              width: 20, height: 20, borderRadius: 4, borderWidth: 2,
                              marginRight: 10, alignItems: 'center', justifyContent: 'center',
                              borderColor: isSelected ? '#2563eb' : '#d1d5db',
                              backgroundColor: isSelected ? '#2563eb' : '#fff',
                            }]}>
                              {isSelected && <Text style={{ color: 'white', fontSize: 12, fontWeight: 'bold' }}>✓</Text>}
                            </View>
                            <View style={{ flex: 1 }}>
                              <Text style={{ fontSize: 13, fontWeight: '600', color: '#1f2937' }}>
                                {student.name}
                              </Text>
                              <Text style={{ fontSize: 11, color: '#6b7280' }}>
                                {student.email || student.roll_no}{student.batch ? ` • ${student.batch}` : ''}
                              </Text>
                            </View>
                          </TouchableOpacity>
                        );
                      })}
                    </ScrollView>
                  </View>
                )}
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
                <Text style={styles.primaryButtonText}>
                  ✓ Create Subject{selectedStudentIds.length > 0 ? ` (${selectedStudentIds.length} students)` : ''}
                </Text>
              )}
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.modalCloseButton, { marginTop: 8 }]}
              onPress={() => { setShowAddSubject(false); setSelectedStudentIds([]); }}
            >
              <Text style={styles.modalCloseButtonText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* ── Update Class Modal ──────────────────────────────────────── */}
      <Modal
        visible={showUpdateClass}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setShowUpdateClass(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { maxHeight: '85%' }]}>
            <Text style={styles.modalTitle}>Update Class</Text>

            {selectedClassData && (
              <View style={{ marginBottom: 12 }}>
                <Text style={{ fontSize: 16, fontWeight: '700', color: '#1f2937' }}>
                  {selectedClassData.name} ({selectedClassData.code})
                </Text>
              </View>
            )}

            <Text style={{ fontWeight: '600', color: '#374151', marginBottom: 4 }}>Batch</Text>
            <TextInput
              style={styles.input}
              placeholder="e.g. 2024"
              value={updateClassBatch}
              onChangeText={setUpdateClassBatch}
            />

            <Text style={{ fontWeight: '600', color: '#374151', marginTop: 8, marginBottom: 4 }}>
              Enrolled Students ({selectedStudentIds.length} selected)
            </Text>

            <ScrollView style={{ maxHeight: 320, borderWidth: 1, borderColor: '#e5e7eb', borderRadius: 8, padding: 4 }}>
              {studentsLoading ? (
                <ActivityIndicator style={{ margin: 24 }} color="#2563eb" />
              ) : availableStudents.length === 0 ? (
                <Text style={{ padding: 16, color: '#9ca3af', textAlign: 'center' }}>
                  No registered students found.
                </Text>
              ) : (
                availableStudents.map((s) => (
                  <TouchableOpacity
                    key={s.id}
                    style={{
                      flexDirection: 'row', alignItems: 'center', paddingVertical: 10, paddingHorizontal: 8,
                      borderBottomWidth: 1, borderBottomColor: '#f3f4f6',
                      backgroundColor: selectedStudentIds.includes(s.id) ? '#eff6ff' : 'transparent',
                    }}
                    onPress={() => toggleStudentSelection(s.id)}
                  >
                    <Ionicons
                      name={selectedStudentIds.includes(s.id) ? 'checkbox' : 'square-outline'}
                      size={22}
                      color={selectedStudentIds.includes(s.id) ? '#2563eb' : '#9ca3af'}
                      style={{ marginRight: 10 }}
                    />
                    <View style={{ flex: 1 }}>
                      <Text style={{ fontWeight: '600', color: '#1f2937' }}>{s.name}</Text>
                      <Text style={{ fontSize: 12, color: '#6b7280' }}>
                        Roll: {s.roll_no || '—'} • {s.batch || '—'}
                      </Text>
                    </View>
                  </TouchableOpacity>
                ))
              )}
            </ScrollView>

            <TouchableOpacity
              style={[styles.primaryButton, { marginTop: 12, opacity: updatingClass ? 0.6 : 1 }]}
              onPress={handleSaveUpdateClass}
              disabled={updatingClass}
            >
              <Text style={styles.primaryButtonText}>
                {updatingClass ? 'Saving...' : '✓ Save Changes'}
              </Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={styles.modalCloseButton}
              onPress={() => setShowUpdateClass(false)}
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

        {/* Action Buttons — 2+2+1 Grid */}
        <View style={{ marginBottom: 16 }}>
          {/* Row 1 */}
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 12 }}>
            <TouchableOpacity
              style={[styles.actionButton, { backgroundColor: '#16a34a', width: '48%' }]}
              onPress={handleStartAttendance}
            >
              <Camera color="white" size={26} style={{ marginBottom: 6 }} />
              <Text style={styles.actionButtonText}>Start{`\n`}Attendance</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.actionButton, { backgroundColor: '#6366f1', width: '48%' }]}
              onPress={() => {
                setSelectedClass(selectedClassData);
                navigateTo('DetailedReport');
              }}
            >
              <BarChart3 color="white" size={26} style={{ marginBottom: 6 }} />
              <Text style={styles.actionButtonText}>View{`\n`}Reports</Text>
            </TouchableOpacity>
          </View>

          {/* Row 2 */}
          <View style={{ flexDirection: 'row', justifyContent: 'space-between', marginBottom: 12 }}>
            <TouchableOpacity
              style={[styles.actionButton, { backgroundColor: '#ea580c', width: '48%' }]}
              onPress={() => navigateTo('RegisterStudent')}
            >
              <Camera color="white" size={26} style={{ marginBottom: 6 }} />
              <Text style={styles.actionButtonText}>Register{`\n`}Student</Text>
            </TouchableOpacity>

            <TouchableOpacity
              style={[styles.actionButton, { backgroundColor: '#0891b2', width: '48%' }]}
              onPress={() => setShowAddSubject(true)}
            >
              <Ionicons name="book-outline" color="white" size={26} style={{ marginBottom: 6 }} />
              <Text style={styles.actionButtonText}>Add{`\n`}Subject</Text>
            </TouchableOpacity>
          </View>

          {/* Row 3 — full width */}
          <TouchableOpacity
            style={[styles.actionButton, { backgroundColor: '#7c3aed', width: '100%', flexDirection: 'row', height: 56, justifyContent: 'center' }]}
            onPress={handleOpenUpdateClass}
          >
            <Settings color="white" size={22} style={{ marginRight: 8 }} />
            <Text style={[styles.actionButtonText, { fontSize: 15 }]}>Update Class</Text>
          </TouchableOpacity>
        </View>


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
        <TouchableOpacity style={styles.navItem} onPress={() => { setSelectedClass(selectedClassData); navigateTo('NotificationHub'); }}>
          <Bell color="#9ca3af" size={24} />
          <Text style={styles.navText}>Notifications</Text>
        </TouchableOpacity>
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
        Alert.alert('Error', scanResult.message || 'No face detected in photo. Please try again.');
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

// --- 5. SCAN ATTENDANCE (PHOTO / VIDEO MODE) ---
const ScanAttendanceScreen = ({ navigateTo, currentSession }) => {
  const [permission, requestPermission] = useCameraPermissions();

  // ── Mode ───────────────────────────────────────────────────────────────────
  const [scanMode, setScanMode] = useState('video');

  // ── UI State ───────────────────────────────────────────────────────────────
  const [isScanning, setIsScanning]         = useState(false);
  const [scanStatus, setScanStatus]         = useState('Ready · Press Start');
  const [statusType, setStatusType]         = useState('idle');
  const [markedStudents, setMarkedStudents] = useState([]);
  const [facing, setFacing]                 = useState('back');

  // ── Queue Stats (live from backend) ───────────────────────────────────────
  const [queueStats, setQueueStats] = useState({ queued: 0, processing: 0, completed: 0, failed: 0 });
  const [socketConnected, setSocketConnected] = useState(false);

  // ── Refs ───────────────────────────────────────────────────────────────────
  const cameraRef       = useRef(null);
  const captureInterval = useRef(null);   // setInterval handle for frame capture loop
  const isScanningRef   = useRef(false);
  const markedIdsRef    = useRef(new Set());
  const socketRef       = useRef(null);   // WebSocket connection handle
  const pollRef         = useRef(null);   // polling interval handle (fallback)

  // ── WebSocket + polling setup ─────────────────────────────────────────────
  useEffect(() => {
    if (!currentSession) return;

    // ── WebSocket (primary) ────────────────────────────────────────────────
    const handleUpdate = (payload) => {
      const { student_id, student_name, present_count, worker } = payload;
      if (!student_id || !student_name) return;

      // UI dedup — only add to list if new
      if (!markedIdsRef.current.has(student_id)) {
        markedIdsRef.current.add(student_id);
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        setMarkedStudents(prev => [{ name: student_name, student_id, time, worker }, ...prev]);
        setScanStatus(`✅ ${student_name} marked present`);
        setStatusType('marked');
      }
    };

    const handleStatus = (stats) => {
      setQueueStats(stats);
    };

    const socket = api.connectSocket(currentSession, handleUpdate, handleStatus);
    socketRef.current = socket;
    setSocketConnected(true);

    // ── Polling fallback (every 1 s) — keeps queue stats updated even if WS ──
    // doesn't fire queue_status events continuously
    pollRef.current = setInterval(async () => {
      const stats = await api.getQueueStatus(currentSession);
      setQueueStats(stats);
      if (stats && Array.isArray(stats.marked_students) && stats.marked_students.length > 0) {
        stats.marked_students.forEach(s => {
          if (s.student_id && s.student_name && !markedIdsRef.current.has(s.student_id)) {
            markedIdsRef.current.add(s.student_id);
            const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            setMarkedStudents(prev => [{ name: s.student_name, student_id: s.student_id, time, worker: 'Sync' }, ...prev]);
            setScanStatus(`✅ ${s.student_name} marked present`);
            setStatusType('marked');
          }
        });
      }
    }, 1000);

    return () => {
      // Cleanup on unmount or session change
      socket.close();
      if (pollRef.current) clearInterval(pollRef.current);
      setSocketConnected(false);
    };
  }, [currentSession]);

  // ── Stop capture loop when mode changes ───────────────────────────────────
  useEffect(() => {
    stopCapture();
    setScanStatus(scanMode === 'video' ? 'Video Mode · Press Start' : 'Photo Mode · Press Capture');
    setStatusType('idle');
  }, [scanMode]);

  // ── Cleanup on unmount ────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      stopCapture();
    };
  }, []);

  // ─────────────────────────────────────────────────────────────────────────
  //  Core: capture one frame and ENQUEUE it (fire-and-forget)
  //  The camera never waits for recognition — it just enqueues and moves on.
  // ─────────────────────────────────────────────────────────────────────────
  const captureAndEnqueue = async () => {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.6,        // optimised for face_recognition (~200-300 KB)
        base64: false,
        skipProcessing: true, // skip camera post-processing for speed
      });
      if (!photo?.uri) return;

      // Fire-and-forget: do NOT await the response before capturing next frame
      api.enqueueFrame(photo.uri, currentSession).then(res => {
        if (res.status === 'queued') {
          setScanStatus('📤 Frame queued — workers scanning…');
          setStatusType('scanning');
        } else if (res.status === 'full') {
          setScanStatus('⚠ Queue full — workers busy');
          setStatusType('error');
        }
      }).catch(() => {/* network error — silent */});

    } catch (err) {
      console.warn('[captureAndEnqueue] Camera error:', err?.message);
    }
  };

  // ─────────────────────────────────────────────────────────────────────────
  //  Start / stop the 700 ms capture interval
  // ─────────────────────────────────────────────────────────────────────────
  const startCapture = () => {
    if (!currentSession) {
      Alert.alert('No Session', 'Go back and start an attendance session first.');
      return;
    }
    isScanningRef.current = true;
    setIsScanning(true);
    setScanStatus('📤 Streaming frames to workers…');
    setStatusType('scanning');

    // Capture first frame immediately
    captureAndEnqueue();

    // Then every 700 ms — no waiting for the API
    captureInterval.current = setInterval(() => {
      if (isScanningRef.current) captureAndEnqueue();
    }, 700);
  };

  const stopCapture = () => {
    isScanningRef.current = false;
    if (captureInterval.current) {
      clearInterval(captureInterval.current);
      captureInterval.current = null;
    }
    setIsScanning(false);
  };

  const pauseCapture = () => {
    stopCapture();
    setScanStatus('Paused');
    setStatusType('idle');
  };

  // ─────────────────────────────────────────────────────────────────────────
  //  Photo mode: single manual capture
  // ─────────────────────────────────────────────────────────────────────────
  const capturePhotoManual = () => {
    if (!currentSession) {
      Alert.alert('No Session', 'Go back and start an attendance session first.');
      return;
    }
    captureAndEnqueue();
  };

  // ─────────────────────────────────────────────────────────────────────────
  //  Stop session
  // ─────────────────────────────────────────────────────────────────────────
  const handleStopSession = async () => {
    pauseCapture();
    try {
      await api.stopSession(currentSession);
      Alert.alert(
        '✅ Session Complete',
        `${markedStudents.length} student${markedStudents.length !== 1 ? 's' : ''} marked present.`,
        [{ text: 'Done', onPress: () => navigateTo('TeacherDashboard') }]
      );
    } catch { Alert.alert('Error', 'Failed to stop session'); }
  };

  // ── Status colour maps ────────────────────────────────────────────────────
  const S_COLOR = { idle: '#6b7280', scanning: '#2563eb', marked: '#059669', error: '#dc2626', unknown: '#d97706' };
  const S_BG    = { idle: '#f3f4f6', scanning: '#eff6ff', marked: '#d1fae5', error: '#fee2e2', unknown: '#fef3c7' };

  // ── Permission guard ──────────────────────────────────────────────────────
  if (!permission) return <View />;
  if (!permission.granted) {
    return (
      <View style={[styles.screenContainer, { justifyContent: 'center', alignItems: 'center', padding: 24 }]}>
        <Text style={{ marginBottom: 16, textAlign: 'center' }}>Camera permission required.</Text>
        <TouchableOpacity style={styles.primaryButton} onPress={requestPermission}>
          <Text style={styles.primaryButtonText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const isVideoMode = scanMode === 'video';

  return (
    <View style={[styles.screenContainer, { backgroundColor: '#0f172a' }]}>

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <View style={[styles.header, { backgroundColor: '#1e293b', borderBottomWidth: 0 }]}>
        <TouchableOpacity onPress={() => navigateTo('TeacherDashboard')} style={{ padding: 10, marginRight: 8 }}>
          <ChevronLeft color="white" size={30} />
        </TouchableOpacity>
        <View style={{ alignItems: 'center', flex: 1 }}>
          <Text style={[styles.headerTitle, { color: 'white' }]}>
            {isVideoMode ? 'Video Attendance' : 'Photo Attendance'}
          </Text>
          <Text style={{ color: '#94a3b8', fontSize: 11, marginTop: 1 }}>
            {isVideoMode
              ? (isScanning ? '🔴 LIVE · Workers processing' : '⏸ PAUSED')
              : '📷 MANUAL CAPTURE'
            }  •  {markedStudents.length} marked present
          </Text>
        </View>
        {/* WebSocket indicator */}
        <View style={{ width: 50, alignItems: 'center' }}>
          <View style={{
            width: 10, height: 10, borderRadius: 5,
            backgroundColor: socketConnected ? '#22c55e' : '#6b7280',
          }} />
          <Text style={{ color: '#64748b', fontSize: 9, marginTop: 2 }}>
            {socketConnected ? 'LIVE' : 'POLL'}
          </Text>
        </View>
      </View>

      {/* ── Mode Toggle ─────────────────────────────────────────────────── */}
      <View style={{
        flexDirection: 'row', marginHorizontal: 16, marginTop: 10, marginBottom: 4,
        backgroundColor: '#1e293b', borderRadius: 12, padding: 4,
      }}>
        <TouchableOpacity
          style={{ flex: 1, paddingVertical: 8, borderRadius: 9, alignItems: 'center', backgroundColor: isVideoMode ? '#2563eb' : 'transparent' }}
          onPress={() => setScanMode('video')}
        >
          <Text style={{ color: isVideoMode ? 'white' : '#64748b', fontWeight: '700', fontSize: 13 }}>📹  Video Mode</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={{ flex: 1, paddingVertical: 8, borderRadius: 9, alignItems: 'center', backgroundColor: !isVideoMode ? '#7c3aed' : 'transparent' }}
          onPress={() => setScanMode('photo')}
        >
          <Text style={{ color: !isVideoMode ? 'white' : '#64748b', fontWeight: '700', fontSize: 13 }}>📷  Photo Mode</Text>
        </TouchableOpacity>
      </View>

      {/* ── Queue Stats Bar ──────────────────────────────────────────────── */}
      <View style={{
        flexDirection: 'row', marginHorizontal: 16, marginBottom: 6,
        backgroundColor: '#1e293b', borderRadius: 10, padding: 8,
        justifyContent: 'space-around',
      }}>
        {[
          { label: 'QUEUED',     value: queueStats.queued,     color: '#60a5fa' },
          { label: 'PROCESSING', value: queueStats.processing, color: '#f59e0b' },
          { label: 'DONE',       value: queueStats.completed,  color: '#22c55e' },
          { label: 'FAILED',     value: queueStats.failed,     color: '#ef4444' },
        ].map(({ label, value, color }) => (
          <View key={label} style={{ alignItems: 'center' }}>
            <Text style={{ color, fontWeight: '800', fontSize: 16 }}>{value}</Text>
            <Text style={{ color: '#475569', fontSize: 9, fontWeight: '600' }}>{label}</Text>
          </View>
        ))}
      </View>

      {/* ── Camera Preview ───────────────────────────────────────────────── */}
      <View style={{
        flex: 1, margin: 12, borderRadius: 20, overflow: 'hidden',
        borderWidth: 2,
        borderColor: isVideoMode ? (isScanning ? '#22c55e' : '#334155') : '#7c3aed',
      }}>
        <CameraView style={{ flex: 1 }} facing={facing} ref={cameraRef}>
          <View style={{ flex: 1, justifyContent: 'center', alignItems: 'center' }}>
            <View style={{
              width: 200, height: 240, borderRadius: 16, borderWidth: 2, borderStyle: 'dashed',
              borderColor: isVideoMode
                ? (isScanning ? 'rgba(34,197,94,0.7)' : 'rgba(255,255,255,0.25)')
                : 'rgba(167,139,250,0.7)',
              justifyContent: 'flex-end', alignItems: 'center', paddingBottom: 12,
            }}>
              <Text style={{ color: 'rgba(255,255,255,0.55)', fontSize: 11 }}>
                {isVideoMode
                  ? (isScanning ? 'Streaming to 3 workers…' : 'Press Start to begin')
                  : 'Tap Capture to scan'}
              </Text>
            </View>
          </View>

          {/* Streaming badge */}
          {isScanning && (
            <View style={{
              position: 'absolute', top: 12, left: 12,
              backgroundColor: 'rgba(37,99,235,0.85)',
              paddingHorizontal: 10, paddingVertical: 5, borderRadius: 8,
              flexDirection: 'row', alignItems: 'center',
            }}>
              <ActivityIndicator size="small" color="white" style={{ marginRight: 6 }} />
              <Text style={{ color: 'white', fontSize: 11, fontWeight: '600' }}>
                700ms · 3 Workers
              </Text>
            </View>
          )}

          {/* Flip button */}
          <TouchableOpacity
            style={{ position: 'absolute', bottom: 14, right: 14, backgroundColor: 'rgba(0,0,0,0.55)', padding: 8, borderRadius: 8 }}
            onPress={() => setFacing(f => f === 'back' ? 'front' : 'back')}
          >
            <Text style={{ color: 'white', fontSize: 12 }}>🔄 Flip</Text>
          </TouchableOpacity>
        </CameraView>
      </View>

      {/* ── Status Badge ─────────────────────────────────────────────────── */}
      <View style={{
        marginHorizontal: 16, marginBottom: 8,
        backgroundColor: S_BG[statusType] || S_BG.idle,
        padding: 10, borderRadius: 10,
        flexDirection: 'row', justifyContent: 'center', alignItems: 'center',
      }}>
        {statusType === 'scanning' && (
          <ActivityIndicator size="small" color={S_COLOR.scanning} style={{ marginRight: 8 }} />
        )}
        <Text style={{ color: S_COLOR[statusType] || S_COLOR.idle, fontWeight: '700', fontSize: 14 }}>
          {scanStatus}
        </Text>
      </View>

      {/* ── Marked Students List ─────────────────────────────────────────── */}
      <View style={{ marginHorizontal: 16, marginBottom: 8, maxHeight: 120 }}>
        <Text style={{ color: '#94a3b8', fontSize: 10, fontWeight: '700', letterSpacing: 1, marginBottom: 6 }}>
          MARKED PRESENT ({markedStudents.length})
        </Text>
        {markedStudents.length === 0
          ? <Text style={{ color: '#475569', fontSize: 12, fontStyle: 'italic' }}>No students marked yet — workers scanning…</Text>
          : <ScrollView showsVerticalScrollIndicator={false} nestedScrollEnabled>
              {markedStudents.map((s, i) => (
                <View key={i} style={{
                  flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
                  backgroundColor: '#1e293b', paddingVertical: 7, paddingHorizontal: 10,
                  borderRadius: 8, marginBottom: 4,
                }}>
                  <View>
                    <Text style={{ color: '#22c55e', fontWeight: '600', fontSize: 13 }}>✅  {s.name}</Text>
                    {s.worker && <Text style={{ color: '#475569', fontSize: 10 }}>{s.worker}</Text>}
                  </View>
                  <Text style={{ color: '#64748b', fontSize: 11 }}>{s.time}</Text>
                </View>
              ))}
            </ScrollView>
        }
      </View>

      {/* ── Action Buttons ───────────────────────────────────────────────── */}
      <View style={{ flexDirection: 'row', gap: 10, paddingHorizontal: 16, paddingBottom: 20 }}>

        {isVideoMode ? (
          !isScanning
            ? <TouchableOpacity
                style={[styles.primaryButton, { flex: 1, backgroundColor: '#16a34a' }]}
                onPress={startCapture}
              >
                <Text style={styles.primaryButtonText}>▶  Start Scanning</Text>
              </TouchableOpacity>
            : <TouchableOpacity
                style={[styles.primaryButton, { flex: 1, backgroundColor: '#d97706' }]}
                onPress={pauseCapture}
              >
                <Text style={styles.primaryButtonText}>⏸  Pause</Text>
              </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={[styles.primaryButton, { flex: 1, backgroundColor: '#7c3aed' }]}
            onPress={capturePhotoManual}
          >
            <Text style={styles.primaryButtonText}>📷  Capture & Scan</Text>
          </TouchableOpacity>
        )}

        <TouchableOpacity
          style={[styles.primaryButton, { flex: 1, backgroundColor: '#dc2626' }]}
          onPress={handleStopSession}
        >
          <Text style={styles.primaryButtonText}>⏹  Stop Session</Text>
        </TouchableOpacity>
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
  const [month, setMonth]   = useState(new Date().getMonth() + 1);
  const [year, setYear]     = useState(new Date().getFullYear());
  const [day, setDay]       = useState(null);   // null = monthly view
  const [report, setReport] = useState(null);
  const [loading, setLoading]       = useState(false);
  const [csvLoading, setCsvLoading] = useState(false);
  const [dayData, setDayData]       = useState(null);   // { date, sessions, students }
  const [dayLoading, setDayLoading] = useState(false);

  // Picker modals
  const [showMonthPicker, setShowMonthPicker] = useState(false);
  const [showYearPicker, setShowYearPicker]   = useState(false);
  const [showDayPicker, setShowDayPicker]     = useState(false);

  // Edit attendance modal state (session-based — unchanged)
  const [editStudent, setEditStudent] = useState(null);
  const [editStatus, setEditStatus]   = useState('Present');
  const [editLoading, setEditLoading] = useState(false);

  // Days available for the selected month/year
  const daysInMonth = new Date(year, month, 0).getDate();
  const DAY_OPTIONS = [
    { label: 'All Dates', value: null },
    ...Array.from({ length: daysInMonth }, (_, i) => ({ label: String(i + 1).padStart(2, '0'), value: i + 1 })),
  ];

  // Load monthly report when class/month/year change
  useEffect(() => {
    if (selectedClass?.id) loadReport(selectedClass.id, month, year);
  }, [selectedClass, month, year]);

  // Load day-specific data when day changes
  useEffect(() => {
    if (day !== null && selectedClass?.id) {
      const dateStr = `${year}-${String(month).padStart(2,'0')}-${String(day).padStart(2,'0')}`;
      loadDayData(selectedClass.id, dateStr);
    } else {
      setDayData(null);
    }
  }, [day, selectedClass, month, year]);

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

  const loadDayData = async (classId, dateStr) => {
    setDayLoading(true);
    try {
      const result = await api.getAttendanceByDate(classId, dateStr);
      setDayData(result.status === 'success' ? result : { date: dateStr, sessions: [], students: [] });
    } catch { setDayData({ date: dateStr, sessions: [], students: [] }); }
    finally { setDayLoading(false); }
  };

  const handleMonthSelect = (m) => { setMonth(m); setDay(null); setDayData(null); setShowMonthPicker(false); };
  const handleYearSelect  = (y) => { setYear(y);  setDay(null); setDayData(null); setShowYearPicker(false); };
  const handleDaySelect   = (d) => { setDay(d); setShowDayPicker(false); };

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
  const selectedDayLabel   = day !== null ? String(day).padStart(2, '0') : 'Day';

  return (
    <View style={styles.screenContainer}>

      {/* ── Month Picker Modal ── */}
      <Modal visible={showMonthPicker} transparent animationType="slide" onRequestClose={() => setShowMonthPicker(false)}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Select Month</Text>
            <ScrollView style={{ maxHeight: 380 }}>
              {MONTHS.map(m => (
                <TouchableOpacity key={m.value} style={[styles.modalItem, m.value === month && styles.modalItemSelected]} onPress={() => handleMonthSelect(m.value)}>
                  <Text style={[styles.modalItemText, m.value === month && { color: '#2563eb' }]}>{m.label}</Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
            <TouchableOpacity style={styles.modalCloseButton} onPress={() => setShowMonthPicker(false)}>
              <Text style={styles.modalCloseButtonText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* ── Year Picker Modal ── */}
      <Modal visible={showYearPicker} transparent animationType="slide" onRequestClose={() => setShowYearPicker(false)}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Select Year</Text>
            <ScrollView style={{ maxHeight: 280 }}>
              {YEARS.map(y => (
                <TouchableOpacity key={y} style={[styles.modalItem, y === year && styles.modalItemSelected]} onPress={() => handleYearSelect(y)}>
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

      {/* ── Day Picker Modal ── */}
      <Modal visible={showDayPicker} transparent animationType="slide" onRequestClose={() => setShowDayPicker(false)}>
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>Select Date</Text>
            <Text style={{ color: '#6b7280', fontSize: 13, marginBottom: 8 }}>{selectedMonthLabel} {year}</Text>
            <ScrollView style={{ maxHeight: 380 }}>
              {DAY_OPTIONS.map(d => (
                <TouchableOpacity
                  key={String(d.value)}
                  style={[styles.modalItem, d.value === day && styles.modalItemSelected]}
                  onPress={() => handleDaySelect(d.value)}
                >
                  <Text style={[styles.modalItemText, d.value === day && { color: '#2563eb' }]}>
                    {d.value === null
                      ? '📅  All Dates (Monthly View)'
                      : `${String(d.value).padStart(2,'0')} ${selectedMonthLabel} ${year}`}
                  </Text>
                </TouchableOpacity>
              ))}
            </ScrollView>
            <TouchableOpacity style={styles.modalCloseButton} onPress={() => setShowDayPicker(false)}>
              <Text style={styles.modalCloseButtonText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* ── Edit Attendance Modal (session-based — unchanged) ── */}
      <Modal visible={!!editStudent} transparent animationType="fade" onRequestClose={() => setEditStudent(null)}>
        <View style={[styles.modalOverlay, { justifyContent: 'center', paddingHorizontal: 24 }]}>
          <View style={[styles.modalContent, { borderRadius: 20 }]}>
            <Text style={styles.modalTitle}>Edit Attendance</Text>
            <Text style={{ color: '#6b7280', fontSize: 14, marginBottom: 16 }}>
              {editStudent?.name}  •  Current: {editStudent?.attendance}%
            </Text>
            <View style={{ flexDirection: 'row', gap: 12, marginBottom: 20 }}>
              <TouchableOpacity style={[{ flex: 1, padding: 16, borderRadius: 12, alignItems: 'center', borderWidth: 2, borderColor: editStatus === 'Present' ? '#16a34a' : '#e5e7eb', backgroundColor: editStatus === 'Present' ? '#dcfce7' : '#f9fafb' }]} onPress={() => setEditStatus('Present')}>
                <Text style={{ fontSize: 22, marginBottom: 4 }}>✅</Text>
                <Text style={{ fontWeight: 'bold', fontSize: 14, color: editStatus === 'Present' ? '#16a34a' : '#9ca3af' }}>Present</Text>
              </TouchableOpacity>
              <TouchableOpacity style={[{ flex: 1, padding: 16, borderRadius: 12, alignItems: 'center', borderWidth: 2, borderColor: editStatus === 'Absent' ? '#dc2626' : '#e5e7eb', backgroundColor: editStatus === 'Absent' ? '#fee2e2' : '#f9fafb' }]} onPress={() => setEditStatus('Absent')}>
                <Text style={{ fontSize: 22, marginBottom: 4 }}>❌</Text>
                <Text style={{ fontWeight: 'bold', fontSize: 14, color: editStatus === 'Absent' ? '#dc2626' : '#9ca3af' }}>Absent</Text>
              </TouchableOpacity>
            </View>
            <TouchableOpacity style={[styles.primaryButton, { marginBottom: 8 }]} onPress={handleSaveEdit} disabled={editLoading}>
              {editLoading ? <ActivityIndicator color="white" /> : <Text style={styles.primaryButtonText}>💾 Save Change</Text>}
            </TouchableOpacity>
            <TouchableOpacity style={styles.modalCloseButton} onPress={() => setEditStudent(null)}>
              <Text style={styles.modalCloseButtonText}>Cancel</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>

      {/* ── Header ── */}
      <View style={styles.header}>
        <TouchableOpacity onPress={() => navigateTo('TeacherDashboard')} style={{ padding: 10, marginRight: 8 }}>
          <ChevronLeft color="#374151" size={30} />
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Attendance Register</Text>
        <TouchableOpacity onPress={handleDownloadCSV} disabled={csvLoading}>
          {csvLoading ? <ActivityIndicator size="small" color="#374151" /> : <Download color="#374151" size={24} />}
        </TouchableOpacity>
      </View>

      <ScrollView style={styles.scrollContent}>
        {/* Class Info */}
        <View style={styles.card}>
          <Text style={styles.sectionTitle}>{selectedClass?.name}</Text>
          <Text style={styles.label}>Total Sessions (this month): {report?.total_classes || 0}</Text>
        </View>

        {/* ── Filter Row — Month · Year · Day ── */}
        <View style={[styles.filterRow, { flexWrap: 'nowrap' }]}>
          <TouchableOpacity style={[styles.filterBox, { flex: 1 }]} onPress={() => setShowMonthPicker(true)}>
            <Text style={{ fontSize: 13, color: '#111827', fontWeight: '500' }} numberOfLines={1}>{selectedMonthLabel}</Text>
            <ChevronDown size={14} color="#6b7280" />
          </TouchableOpacity>

          <TouchableOpacity style={[styles.filterBox, { width: 70 }]} onPress={() => setShowYearPicker(true)}>
            <Text style={{ fontSize: 13, color: '#111827', fontWeight: '500' }}>{year}</Text>
            <ChevronDown size={14} color="#6b7280" />
          </TouchableOpacity>

          {/* Day Picker — highlighted blue when a day is selected */}
          <TouchableOpacity
            style={[styles.filterBox, { width: 80, borderColor: day !== null ? '#2563eb' : '#e5e7eb', backgroundColor: day !== null ? '#eff6ff' : 'white' }]}
            onPress={() => setShowDayPicker(true)}
          >
            <Text style={{ fontSize: 13, color: day !== null ? '#2563eb' : '#6b7280', fontWeight: day !== null ? '700' : '500' }}>
              {day !== null ? `📅 ${selectedDayLabel}` : 'Day'}
            </Text>
            <ChevronDown size={14} color={day !== null ? '#2563eb' : '#6b7280'} />
          </TouchableOpacity>
        </View>

        {/* ── DAY VIEW (when a specific date is selected) ── */}
        {day !== null && (
          <>
            {/* Date header card */}
            <View style={[styles.card, { backgroundColor: '#eff6ff', borderColor: '#bfdbfe' }]}>
              <Text style={{ fontSize: 15, fontWeight: '700', color: '#1d4ed8', textAlign: 'center' }}>
                📅  {selectedDayLabel} {selectedMonthLabel} {year}
              </Text>
              {dayData?.sessions?.length > 0 && (
                <Text style={{ textAlign: 'center', color: '#3b82f6', fontSize: 12, marginTop: 4 }}>
                  {dayData.sessions.length} session{dayData.sessions.length !== 1 ? 's' : ''} held  •  {dayData.sessions.reduce((s, x) => s + (x.total_scanned || 0), 0)} total scanned
                </Text>
              )}
            </View>

            {dayLoading ? (
              <ActivityIndicator size="large" color="#2563eb" style={{ marginTop: 40 }} />
            ) : !dayData || dayData.students.length === 0 ? (
              <View style={[styles.card, { alignItems: 'center', padding: 36 }]}>
                <Text style={{ fontSize: 30, marginBottom: 8 }}>📭</Text>
                <Text style={{ color: '#374151', fontSize: 15, fontWeight: '600' }}>No data for this date</Text>
                <Text style={{ color: '#9ca3af', fontSize: 12, marginTop: 4, textAlign: 'center' }}>
                  No session was held on {selectedDayLabel} {selectedMonthLabel} {year}
                </Text>
              </View>
            ) : (
              <View style={styles.card}>
                {/* Present / Absent summary */}
                <View style={[styles.statsRow, { marginBottom: 12 }]}>
                  <View style={[styles.miniStat, { backgroundColor: '#dcfce7' }]}>
                    <Text style={[styles.miniStatLabel, { color: '#16a34a' }]}>PRESENT</Text>
                    <Text style={[styles.miniStatValue, { color: '#16a34a' }]}>{dayData.students.filter(s => s.status === 'Present').length}</Text>
                  </View>
                  <View style={[styles.miniStat, { backgroundColor: '#fee2e2' }]}>
                    <Text style={[styles.miniStatLabel, { color: '#dc2626' }]}>ABSENT</Text>
                    <Text style={[styles.miniStatValue, { color: '#dc2626' }]}>{dayData.students.filter(s => s.status === 'Absent').length}</Text>
                  </View>
                  <View style={[styles.miniStat, { backgroundColor: '#dbeafe' }]}>
                    <Text style={[styles.miniStatLabel, { color: '#2563eb' }]}>TOTAL</Text>
                    <Text style={[styles.miniStatValue, { color: '#2563eb' }]}>{dayData.students.length}</Text>
                  </View>
                </View>

                {/* Per-student status */}
                {dayData.students.map((student, index) => (
                  <View key={index} style={[styles.listItem, { backgroundColor: index % 2 === 0 ? '#ffffff' : '#f9fafb', paddingHorizontal: 8, borderRadius: 8 }]}>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.listName}>{student.name}</Text>
                      <Text style={styles.listSub}>Roll: {student.roll_no}</Text>
                    </View>
                    <View style={{ paddingHorizontal: 12, paddingVertical: 5, borderRadius: 8, backgroundColor: student.status === 'Present' ? '#dcfce7' : '#fee2e2' }}>
                      <Text style={{ fontWeight: '700', fontSize: 13, color: student.status === 'Present' ? '#16a34a' : '#dc2626' }}>
                        {student.status === 'Present' ? '✅ Present' : '❌ Absent'}
                      </Text>
                    </View>
                  </View>
                ))}
              </View>
            )}
          </>
        )}

        {/* ── MONTHLY VIEW (default — no day selected) ── */}
        {day === null && (
          <>
            {/* CSV Download Banner */}
            <TouchableOpacity
              style={{ flexDirection: 'row', alignItems: 'center', justifyContent: 'center', backgroundColor: '#eff6ff', padding: 10, borderRadius: 10, marginBottom: 12, borderWidth: 1, borderColor: '#bfdbfe' }}
              onPress={handleDownloadCSV} disabled={csvLoading}
            >
              <Download color="#2563eb" size={16} style={{ marginRight: 8 }} />
              <Text style={{ color: '#2563eb', fontWeight: '600', fontSize: 13 }}>
                {csvLoading ? 'Preparing CSV…' : `Download CSV — ${selectedMonthLabel} ${year}`}
              </Text>
            </TouchableOpacity>

            {loading ? (
              <ActivityIndicator size="large" color="#2563eb" style={{ marginTop: 40 }} />
            ) : !report?.students?.length ? (
              <View style={[styles.card, { alignItems: 'center', padding: 32 }]}>
                <Text style={{ color: '#6b7280', fontSize: 14 }}>No sessions found for {selectedMonthLabel} {year}</Text>
              </View>
            ) : (
              <View style={styles.card}>
                <View style={[styles.cardHeader, { marginBottom: 12 }]}>
                  <Text style={styles.sectionTitle}>Students ({report.students.length})</Text>
                  <Text style={styles.label}>{report.total_classes} sessions</Text>
                </View>
                {report.students.map((student, index) => (
                  <View key={index} style={[styles.listItem, { backgroundColor: index % 2 === 0 ? '#ffffff' : '#f9fafb', paddingHorizontal: 8, borderRadius: 8 }]}>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.listName}>{student.name}</Text>
                      <Text style={styles.listSub}>Roll: {student.roll_no}  •  {student.present}/{student.total} present</Text>
                    </View>
                    <View style={{ alignItems: 'flex-end' }}>
                      <Text style={[styles.listScore, { color: student.attendance >= 75 ? '#16a34a' : '#dc2626' }]}>{student.attendance}%</Text>
                      <TouchableOpacity
                        style={{ marginTop: 4, paddingHorizontal: 10, paddingVertical: 3, borderRadius: 6, borderWidth: 1, borderColor: '#2563eb', backgroundColor: '#eff6ff' }}
                        onPress={() => openEditModal(student)}
                      >
                        <Text style={{ color: '#2563eb', fontSize: 11, fontWeight: '700' }}>✏ Edit</Text>
                      </TouchableOpacity>
                    </View>
                  </View>
                ))}
              </View>
            )}
          </>
        )}
      </ScrollView>
    </View>
  );
};


// --- 6. NOTIFICATION HUB ---
const NotificationHubScreen = ({ navigateTo, selectedClass }) => {
  const [target, setTarget] = useState('defaulters');
  const [message, setMessage] = useState('Dear Student, your attendance is below the required threshold. Please meet the HOD immediately.');
  const [loading, setLoading] = useState(false);
  const [classStudents, setClassStudents] = useState([]);
  const [selectedStudentIds, setSelectedStudentIds] = useState([]);
  const [studentsLoading, setStudentsLoading] = useState(false);

  // Load enrolled students for this class
  useEffect(() => {
    const loadEnrolledStudents = async () => {
      if (!selectedClass?.students || selectedClass.students.length === 0) {
        setClassStudents([]);
        return;
      }
      setStudentsLoading(true);
      try {
        const res = await api.getStudents();
        if (res.status === 'success') {
          // Filter to only enrolled students
          const enrolledIds = new Set(selectedClass.students);
          const enrolled = (res.data || []).filter(s => enrolledIds.has(s.id));
          setClassStudents(enrolled);
        }
      } catch {
        // silent
      } finally {
        setStudentsLoading(false);
      }
    };
    loadEnrolledStudents();
  }, [selectedClass]);

  const toggleStudentSelection = (id) => {
    setSelectedStudentIds(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };

  const handleSend = async () => {
    if (!selectedClass?.id) {
      Alert.alert('Error', 'No class selected. Go back and select a class.');
      return;
    }
    if (!message.trim()) {
      Alert.alert('Error', 'Notification message cannot be empty.');
      return;
    }
    if (target === 'individual' && selectedStudentIds.length === 0) {
      Alert.alert('Error', 'Please select at least one student.');
      return;
    }

    setLoading(true);
    try {
      const result = await api.sendNotification(
        selectedClass.id,
        target,
        message,
        null,  // email (legacy)
        target === 'individual' ? selectedStudentIds : null
      );
      if (result.status === 'success') {
        Alert.alert('Success', result.message || 'Notifications sent successfully!');
        setSelectedStudentIds([]);
      } else {
        Alert.alert('Error', result.message || 'Failed to send notifications.');
      }
    } catch (error) {
      Alert.alert('Error', error.message || 'Network error.');
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
        <Text style={styles.headerTitle}>Notification Hub</Text>
        <View style={{ width: 24 }} />
      </View>

      <ScrollView style={styles.scrollContent}>
        {/* Class context */}
        {selectedClass && (
          <View style={{ marginBottom: 12, padding: 12, backgroundColor: '#eff6ff', borderRadius: 12 }}>
            <Text style={{ fontWeight: '700', color: '#1f2937', fontSize: 15 }}>
              {selectedClass.name} ({selectedClass.code})
            </Text>
            <Text style={{ color: '#6b7280', fontSize: 12, marginTop: 2 }}>
              {selectedClass.total_students || 0} enrolled students
            </Text>
          </View>
        )}

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
              style={[styles.gridItem, target === 'individual' && { backgroundColor: '#f3f4f6', borderColor: '#374151', borderWidth: 2 }]}
              onPress={() => setTarget('individual')}
            >
              <Text style={{ color: '#374151', fontWeight: 'bold' }}>👤 Individual</Text>
              <Text style={{ fontSize: 10, color: '#6b7280' }}>Select Students</Text>
            </TouchableOpacity>
          </View>

          {/* Individual: multi-select student checkboxes */}
          {target === 'individual' && (
            <View style={{ marginTop: 12 }}>
              <Text style={{ fontWeight: '600', color: '#374151', marginBottom: 6 }}>
                INDIVIDUAL STUDENTS
              </Text>

              <ScrollView style={{ maxHeight: 280, borderWidth: 1, borderColor: '#e5e7eb', borderRadius: 8, padding: 4 }}>
                {studentsLoading ? (
                  <ActivityIndicator style={{ margin: 24 }} color="#2563eb" />
                ) : classStudents.length === 0 ? (
                  <Text style={{ padding: 16, color: '#9ca3af', textAlign: 'center' }}>
                    No students are enrolled in this class.
                  </Text>
                ) : (
                  classStudents.map((s) => (
                    <TouchableOpacity
                      key={s.id}
                      style={{
                        flexDirection: 'row', alignItems: 'center', paddingVertical: 10, paddingHorizontal: 8,
                        borderBottomWidth: 1, borderBottomColor: '#f3f4f6',
                        backgroundColor: selectedStudentIds.includes(s.id) ? '#eff6ff' : 'transparent',
                      }}
                      onPress={() => toggleStudentSelection(s.id)}
                    >
                      <Ionicons
                        name={selectedStudentIds.includes(s.id) ? 'checkbox' : 'square-outline'}
                        size={22}
                        color={selectedStudentIds.includes(s.id) ? '#2563eb' : '#9ca3af'}
                        style={{ marginRight: 10 }}
                      />
                      <View style={{ flex: 1 }}>
                        <Text style={{ fontWeight: '600', color: '#1f2937' }}>{s.name}</Text>
                        <Text style={{ fontSize: 12, color: '#6b7280' }}>
                          {s.email || '—'}{s.roll_no ? ` • ${s.roll_no}` : ''}
                        </Text>
                      </View>
                    </TouchableOpacity>
                  ))
                )}
              </ScrollView>

              <Text style={{ textAlign: 'right', color: '#6b7280', fontSize: 12, marginTop: 4 }}>
                Selected: {selectedStudentIds.length} student{selectedStudentIds.length !== 1 ? 's' : ''}
              </Text>
            </View>
          )}
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
          style={[styles.primaryButton, { backgroundColor: '#4f46e5', opacity: loading ? 0.6 : 1 }]}
          onPress={handleSend}
          disabled={loading}
        >
          {loading ? (
            <ActivityIndicator color="white" />
          ) : (
            <Text style={styles.primaryButtonText}>📤 Send Notification</Text>
          )}
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
  actionButton: { padding: 16, borderRadius: 16, alignItems: 'center', justifyContent: 'center', minHeight: 100 },
  actionButtonText: { color: 'white', fontWeight: 'bold', textAlign: 'center', fontSize: 14 },

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