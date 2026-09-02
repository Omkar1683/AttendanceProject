/**
 * student-module/screens/StudentProfile.js
 * -------------------------------------------
 * Feature 8: Profile viewing and editing + password change.
 */
import React, { useState, useEffect, useCallback } from 'react';
import {
  View, Text, ScrollView, StyleSheet, TouchableOpacity,
  TextInput, Alert, ActivityIndicator, RefreshControl,
} from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { Colors, FontSizes, Spacing, BorderRadius } from '../constants/theme';
import { DashboardSkeleton } from '../components/LoadingSkeleton';
import ErrorState from '../components/ErrorState';
import studentApi from '../services/studentApi';

const StudentProfile = ({ userInfo, onLogout, navigateTo }) => {
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState(null);

  // Edit mode
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState('');
  const [editPhone, setEditPhone] = useState('');
  const [editDept, setEditDept] = useState('');
  const [saving, setSaving] = useState(false);

  // Password change
  const [showPasswordChange, setShowPasswordChange] = useState(false);
  const [oldPassword, setOldPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [changingPassword, setChangingPassword] = useState(false);

  const loadProfile = useCallback(async () => {
    try {
      setError(null);
      const res = await studentApi.getProfile();
      if (res.status === 'success') {
        setProfile(res.data);
        setEditName(res.data.name || '');
        setEditPhone(res.data.phone || '');
        setEditDept(res.data.department || '');
      }
    } catch (err) {
      setError(err.message || 'Failed to load profile');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { loadProfile(); }, [loadProfile]);

  const onRefresh = useCallback(() => { setRefreshing(true); loadProfile(); }, [loadProfile]);

  const handleSave = async () => {
    if (!editName.trim()) {
      Alert.alert('Error', 'Name is required');
      return;
    }
    setSaving(true);
    try {
      const res = await studentApi.updateProfile({
        name: editName.trim(),
        phone: editPhone.trim(),
        department: editDept.trim(),
      });
      if (res.status === 'success') {
        Alert.alert('Success', 'Profile updated!');
        setEditing(false);
        loadProfile();
      } else {
        Alert.alert('Error', res.message || 'Update failed');
      }
    } catch (err) {
      Alert.alert('Error', err.message || 'Network error');
    } finally {
      setSaving(false);
    }
  };

  const handleChangePassword = async () => {
    if (!oldPassword || !newPassword) {
      Alert.alert('Error', 'All fields are required');
      return;
    }
    if (newPassword.length < 8) {
      Alert.alert('Error', 'New password must be at least 8 characters');
      return;
    }
    if (newPassword !== confirmPassword) {
      Alert.alert('Error', 'Passwords do not match');
      return;
    }

    setChangingPassword(true);
    try {
      const res = await studentApi.changePassword(oldPassword, newPassword);
      if (res.status === 'success') {
        Alert.alert('Success', 'Password changed successfully!');
        setShowPasswordChange(false);
        setOldPassword('');
        setNewPassword('');
        setConfirmPassword('');
      } else {
        Alert.alert('Error', res.message || 'Password change failed');
      }
    } catch (err) {
      Alert.alert('Error', err.message || 'Network error');
    } finally {
      setChangingPassword(false);
    }
  };

  if (loading) return <DashboardSkeleton />;
  if (error) return <ErrorState message={error} onRetry={loadProfile} />;

  return (
    <ScrollView
      style={profStyles.container}
      contentContainerStyle={profStyles.content}
      refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} colors={[Colors.primary]} />}
      showsVerticalScrollIndicator={false}
    >
      {/* Avatar Card */}
      <View style={profStyles.avatarCard}>
        <View style={profStyles.avatarCircle}>
          <Text style={profStyles.avatarText}>
            {(profile?.name || 'S').charAt(0).toUpperCase()}
          </Text>
        </View>
        <Text style={profStyles.profileName}>{profile?.name || 'Student'}</Text>
        <Text style={profStyles.profileEmail}>{profile?.email || ''}</Text>
      </View>

      {/* Profile Info */}
      <View style={profStyles.card}>
        <View style={profStyles.cardHeader}>
          <Text style={profStyles.cardLabel}>PERSONAL INFORMATION</Text>
          {!editing && (
            <TouchableOpacity onPress={() => setEditing(true)}>
              <Text style={profStyles.editLink}>Edit</Text>
            </TouchableOpacity>
          )}
        </View>

        {editing ? (
          <>
            <View style={profStyles.fieldGroup}>
              <Text style={profStyles.fieldLabel}>Full Name</Text>
              <TextInput
                style={profStyles.input}
                value={editName}
                onChangeText={setEditName}
                placeholder="Your name"
                placeholderTextColor={Colors.textMuted}
              />
            </View>
            <View style={profStyles.fieldGroup}>
              <Text style={profStyles.fieldLabel}>Phone</Text>
              <TextInput
                style={profStyles.input}
                value={editPhone}
                onChangeText={setEditPhone}
                placeholder="Phone number"
                placeholderTextColor={Colors.textMuted}
                keyboardType="phone-pad"
              />
            </View>
            <View style={profStyles.fieldGroup}>
              <Text style={profStyles.fieldLabel}>Department</Text>
              <TextInput
                style={profStyles.input}
                value={editDept}
                onChangeText={setEditDept}
                placeholder="Department"
                placeholderTextColor={Colors.textMuted}
              />
            </View>
            <View style={profStyles.buttonRow}>
              <TouchableOpacity
                style={profStyles.cancelBtn}
                onPress={() => { setEditing(false); setEditName(profile?.name || ''); setEditPhone(profile?.phone || ''); setEditDept(profile?.department || ''); }}
              >
                <Text style={profStyles.cancelText}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity style={profStyles.saveBtn} onPress={handleSave} disabled={saving}>
                {saving ? <ActivityIndicator size="small" color={Colors.textWhite} /> : (
                  <Text style={profStyles.saveText}>Save Changes</Text>
                )}
              </TouchableOpacity>
            </View>
          </>
        ) : (
          <>
            <ProfileField label="Full Name" value={profile?.name} />
            <ProfileField label="Roll No" value={profile?.roll_no} />
            <ProfileField label="Email" value={profile?.email} />
            <ProfileField label="Phone" value={profile?.phone || 'Not set'} />
            <ProfileField label="Department" value={profile?.department || 'Not set'} />
            <ProfileField label="Batch" value={profile?.batch || 'Not set'} />
          </>
        )}
      </View>

      {/* Password Change */}
      <View style={profStyles.card}>
        <TouchableOpacity
          style={profStyles.cardHeader}
          onPress={() => setShowPasswordChange(!showPasswordChange)}
        >
          <Text style={profStyles.cardLabel}>CHANGE PASSWORD</Text>
          <Ionicons
            name={showPasswordChange ? 'chevron-up' : 'chevron-down'}
            size={16} color={Colors.textMuted}
          />
        </TouchableOpacity>

        {showPasswordChange && (
          <>
            <View style={profStyles.fieldGroup}>
              <Text style={profStyles.fieldLabel}>Current Password</Text>
              <TextInput
                style={profStyles.input}
                value={oldPassword}
                onChangeText={setOldPassword}
                placeholder="••••••••"
                placeholderTextColor={Colors.textMuted}
                secureTextEntry
              />
            </View>
            <View style={profStyles.fieldGroup}>
              <Text style={profStyles.fieldLabel}>New Password (min 8 chars)</Text>
              <TextInput
                style={profStyles.input}
                value={newPassword}
                onChangeText={setNewPassword}
                placeholder="••••••••"
                placeholderTextColor={Colors.textMuted}
                secureTextEntry
              />
            </View>
            <View style={profStyles.fieldGroup}>
              <Text style={profStyles.fieldLabel}>Confirm New Password</Text>
              <TextInput
                style={profStyles.input}
                value={confirmPassword}
                onChangeText={setConfirmPassword}
                placeholder="••••••••"
                placeholderTextColor={Colors.textMuted}
                secureTextEntry
              />
            </View>
            <TouchableOpacity
              style={profStyles.saveBtn}
              onPress={handleChangePassword}
              disabled={changingPassword}
            >
              {changingPassword ? <ActivityIndicator size="small" color={Colors.textWhite} /> : (
                <Text style={profStyles.saveText}>Update Password</Text>
              )}
            </TouchableOpacity>
          </>
        )}
      </View>

      {/* Actions */}
      <View style={profStyles.card}>
        <TouchableOpacity
          style={profStyles.actionRow}
          onPress={() => navigateTo('ExportAttendance')}
        >
          <Ionicons name="download-outline" size={20} color={Colors.primary} />
          <Text style={profStyles.actionText}>Export Attendance</Text>
          <Ionicons name="chevron-forward" size={16} color={Colors.textMuted} />
        </TouchableOpacity>
      </View>

      {/* Logout */}
      <TouchableOpacity style={profStyles.logoutBtn} onPress={onLogout}>
        <Ionicons name="log-out-outline" size={20} color={Colors.danger} />
        <Text style={profStyles.logoutText}>Logout</Text>
      </TouchableOpacity>

      <View style={{ height: 24 }} />
    </ScrollView>
  );
};

const ProfileField = ({ label, value }) => (
  <View style={profStyles.field}>
    <Text style={profStyles.fieldLabel}>{label}</Text>
    <Text style={profStyles.fieldValue}>{value || '—'}</Text>
  </View>
);

const profStyles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  content: { padding: Spacing.lg },
  // Avatar
  avatarCard: {
    backgroundColor: Colors.accent, borderRadius: BorderRadius.xxl,
    padding: Spacing.xxl, alignItems: 'center', marginBottom: Spacing.lg,
  },
  avatarCircle: {
    width: 72, height: 72, borderRadius: 36,
    backgroundColor: 'rgba(255,255,255,0.2)',
    alignItems: 'center', justifyContent: 'center',
    borderWidth: 2, borderColor: 'rgba(255,255,255,0.3)',
    marginBottom: Spacing.md,
  },
  avatarText: { color: Colors.textWhite, fontSize: 28, fontWeight: 'bold' },
  profileName: { color: Colors.textWhite, fontSize: FontSizes.xxl, fontWeight: 'bold' },
  profileEmail: { color: 'rgba(255,255,255,0.7)', fontSize: FontSizes.sm, marginTop: 4 },
  // Card
  card: {
    backgroundColor: Colors.surface, borderRadius: BorderRadius.xl,
    padding: Spacing.lg, marginBottom: Spacing.lg,
    borderWidth: 1, borderColor: Colors.border,
  },
  cardHeader: {
    flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center',
    marginBottom: Spacing.md,
  },
  cardLabel: {
    fontSize: FontSizes.xs, fontWeight: 'bold', color: Colors.textMuted, letterSpacing: 1,
  },
  editLink: { fontSize: FontSizes.sm, color: Colors.primary, fontWeight: '600' },
  // Fields
  field: {
    paddingVertical: Spacing.md, borderBottomWidth: 1, borderBottomColor: Colors.borderLight,
  },
  fieldGroup: { marginBottom: Spacing.md },
  fieldLabel: { fontSize: FontSizes.xs, color: Colors.textMuted, fontWeight: '600', marginBottom: 4 },
  fieldValue: { fontSize: FontSizes.md, color: Colors.text, fontWeight: '500' },
  // Input
  input: {
    backgroundColor: Colors.surfaceAlt, borderWidth: 1, borderColor: Colors.border,
    borderRadius: BorderRadius.md, padding: Spacing.md,
    fontSize: FontSizes.md, color: Colors.text,
  },
  // Buttons
  buttonRow: { flexDirection: 'row', gap: Spacing.sm, marginTop: Spacing.md },
  cancelBtn: {
    flex: 1, padding: Spacing.md, borderRadius: BorderRadius.lg,
    backgroundColor: Colors.surfaceAlt, alignItems: 'center',
  },
  cancelText: { color: Colors.textSecondary, fontWeight: '600' },
  saveBtn: {
    flex: 1, padding: Spacing.md, borderRadius: BorderRadius.lg,
    backgroundColor: Colors.primary, alignItems: 'center',
  },
  saveText: { color: Colors.textWhite, fontWeight: '600' },
  // Actions
  actionRow: {
    flexDirection: 'row', alignItems: 'center', paddingVertical: Spacing.sm,
  },
  actionText: {
    flex: 1, marginLeft: Spacing.md, fontSize: FontSizes.md,
    color: Colors.text, fontWeight: '500',
  },
  // Logout
  logoutBtn: {
    flexDirection: 'row', alignItems: 'center', justifyContent: 'center',
    padding: Spacing.lg, backgroundColor: Colors.dangerBg,
    borderRadius: BorderRadius.xl, borderWidth: 1, borderColor: Colors.dangerLight,
  },
  logoutText: {
    color: Colors.danger, fontWeight: 'bold', fontSize: FontSizes.md,
    marginLeft: Spacing.sm,
  },
});

export default StudentProfile;
