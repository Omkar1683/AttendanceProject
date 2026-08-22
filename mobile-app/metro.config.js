const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

const config = getDefaultConfig(__dirname);

// Allow Metro to resolve modules from the student-module directory
// which sits outside the mobile-app project root
const studentModulePath = path.resolve(__dirname, '../student-module');

config.watchFolders = [studentModulePath];

// Ensure node_modules from mobile-app is used for shared dependencies
config.resolver.nodeModulesPaths = [
  path.resolve(__dirname, 'node_modules'),
];

module.exports = config;
