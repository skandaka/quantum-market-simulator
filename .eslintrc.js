// File path: /Users/skandaa/[your-project-root]/.eslintrc.js
module.exports = {
    root: true,
    parser: '@typescript-eslint/parser',
    parserOptions: {
        ecmaVersion: 2020,
        sourceType: 'module',
        ecmaFeatures: {
            jsx: true
        }
    },
    settings: {
        react: {
            version: 'detect'
        }
    },
    env: {
        browser: true,
        node: true,
        es6: true
    },
    extends: [
        'eslint:recommended',
        'plugin:@typescript-eslint/recommended',
        'plugin:react/recommended',
        'plugin:react-hooks/recommended'
    ],
    plugins: [
        '@typescript-eslint',
        'react',
        'react-hooks'
    ],
    rules: {
        // TypeScript specific rules
        '@typescript-eslint/explicit-module-boundary-types': 'off',
        '@typescript-eslint/no-explicit-any': 'warn',
        '@typescript-eslint/no-unused-vars': ['error', {
            argsIgnorePattern: '^_',
            varsIgnorePattern: '^_'
        }],

        // React specific rules
        'react/prop-types': 'off',
        'react/react-in-jsx-scope': 'off',

        // General rules
        'no-console': ['warn', { allow: ['warn', 'error'] }],
        'no-debugger': 'error',
        'semi': ['error', 'always'],
        'quotes': ['error', 'single', { avoidEscape: true }],
        'comma-dangle': ['error', 'never'],
        'indent': ['error', 2],
        'no-multiple-empty-lines': ['error', { max: 1 }],
        'no-trailing-spaces': 'error',
        'object-curly-spacing': ['error', 'always'],
        'array-bracket-spacing': ['error', 'never'],

        // Arrow function rules
        'arrow-body-style': ['error', 'as-needed'],
        'arrow-parens': ['error', 'as-needed'],

        // Import rules
        'import/prefer-default-export': 'off',
        'import/no-unresolved': 'off',

        // Accessibility
        'jsx-a11y/click-events-have-key-events': 'off',
        'jsx-a11y/no-static-element-interactions': 'off'
    },
    overrides: [
        {
            files: ['*.ts', '*.tsx'],
            rules: {
                // TypeScript files specific rules
            }
        }
    ]
};