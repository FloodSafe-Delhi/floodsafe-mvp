# FloodSafe Authentication Setup Guide

This guide will walk you through setting up Google OAuth and Firebase Phone Authentication for your FloodSafe application.

**Your Firebase Project ID**: `gen-lang-client-0669818939`

---

## Part 1: Firebase Console Setup

### Step 1: Access Your Firebase Project

1. Go to [Firebase Console](https://console.firebase.google.com)
2. Click on your existing project: **gen-lang-client-0669818939**

### Step 2: Enable Phone Authentication

1. In the left sidebar, click **Build** → **Authentication**
2. Click the **Get Started** button (if first time)
3. Click on the **Sign-in method** tab
4. Find **Phone** in the list of providers
5. Click on **Phone** to expand it
6. Toggle the **Enable** switch to ON
7. Click **Save**

### Step 3: Add Authorized Domains

1. Still in **Authentication** → **Sign-in method**
2. Scroll down to **Authorized domains**
3. Click **Add domain**
4. Add: `localhost` (for development)
5. Later, add your production domain when you deploy

### Step 4: Get Firebase Configuration

1. Click the ⚙️ gear icon next to **Project Overview** in the left sidebar
2. Click **Project settings**
3. Scroll down to **Your apps** section
4. If you don't have a web app yet:
   - Click the **</>** (Web) icon
   - Register app with nickname: "FloodSafe Web"
   - Check "Also set up Firebase Hosting" (optional)
   - Click **Register app**
5. You'll see your Firebase config object with these values:
   ```javascript
   const firebaseConfig = {
     apiKey: "AIza...",
     authDomain: "gen-lang-client-0669818939.firebaseapp.com",
     projectId: "gen-lang-client-0669818939",
     storageBucket: "gen-lang-client-0669818939.appspot.com",
     messagingSenderId: "123456789",
     appId: "1:123456789:web:abc123..."
   };
   ```
6. **Copy these values** - you'll need them for the frontend `.env` file

---

## Part 2: Google Cloud Console Setup (for Google OAuth)

### Step 5: Access Google Cloud Console

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. In the top dropdown, select your Firebase project: **gen-lang-client-0669818939**
   - Firebase projects are automatically linked to Google Cloud

### Step 6: Enable Google Identity Services

1. In the left sidebar, click **APIs & Services** → **Library**
2. Search for "Google Identity"
3. Click on **Google Identity Toolkit API**
4. Click **Enable** (if not already enabled)

### Step 7: Configure OAuth Consent Screen

1. Go to **APIs & Services** → **OAuth consent screen**
2. Select **External** user type (for testing)
3. Click **Create**
4. Fill in the required fields:
   - **App name**: FloodSafe
   - **User support email**: Your email
   - **Developer contact email**: Your email
5. Click **Save and Continue**
6. On **Scopes** page, click **Save and Continue** (default scopes are fine)
7. On **Test users** page, add your email as a test user
8. Click **Save and Continue**
9. Review and click **Back to Dashboard**

### Step 8: Create OAuth 2.0 Credentials

1. Go to **APIs & Services** → **Credentials**
2. Click **+ Create Credentials** → **OAuth client ID**
3. Select **Application type**: Web application
4. **Name**: FloodSafe Web Client
5. Under **Authorized JavaScript origins**, add:
   - `http://localhost:5175`
6. Under **Authorized redirect URIs**, add:
   - `http://localhost:5175`
7. Click **Create**
8. A dialog will show your **Client ID** and **Client Secret**
9. **Copy the Client ID** - it looks like: `123456789-abc123.apps.googleusercontent.com`
10. **Copy the Client Secret** (you'll need this for backend)

---

## Part 3: Configure FloodSafe Backend

### Step 9: Create Backend .env File

1. Navigate to: `c:\Users\Anirudh Mohan\Desktop\FloodSafe\apps\backend\`
2. Copy `.env.example` to `.env`
3. Open `.env` and fill in:

```bash
# Database (keep your existing value)
DATABASE_URL=postgresql://user:password@localhost:5432/floodsafe

# JWT Authentication (CHANGE THIS IN PRODUCTION!)
JWT_SECRET_KEY=floodsafe-jwt-secret-change-in-production-min-32-chars
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7

# Google OAuth (from Step 8)
GOOGLE_CLIENT_ID=123456789-abc123.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-abc123xyz...

# Firebase (from Step 4)
FIREBASE_PROJECT_ID=gen-lang-client-0669818939
```

### Step 10: Run Database Migration

Open a terminal in the backend directory and run:

```bash
cd "c:\Users\Anirudh Mohan\Desktop\FloodSafe\apps\backend"
python -m src.scripts.migrate_add_auth_fields
```

This will add the new authentication fields to your database.

---

## Part 4: Configure FloodSafe Frontend

### Step 11: Create Frontend .env File

1. Navigate to: `c:\Users\Anirudh Mohan\Desktop\FloodSafe\apps\frontend\`
2. Copy `.env.example` to `.env`
3. Open `.env` and fill in:

```bash
# Backend API URL (without /api suffix)
VITE_API_URL=http://localhost:8000

# Google OAuth (from Step 8 - Client ID only)
VITE_GOOGLE_CLIENT_ID=123456789-abc123.apps.googleusercontent.com

# Firebase Configuration (from Step 4)
VITE_FIREBASE_API_KEY=AIza...
VITE_FIREBASE_AUTH_DOMAIN=gen-lang-client-0669818939.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=gen-lang-client-0669818939
VITE_FIREBASE_STORAGE_BUCKET=gen-lang-client-0669818939.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=123456789
VITE_FIREBASE_APP_ID=1:123456789:web:abc123...
```

---

## Part 5: Test the Setup

### Step 12: Restart Docker Containers

```bash
cd "c:\Users\Anirudh Mohan\Desktop\FloodSafe"
docker-compose down
docker-compose up --build
```

### Step 13: Test Authentication

1. Open your browser to `http://localhost:5175`
2. You should see the **LoginScreen** with two tabs:
   - **Google**: Click the Google Sign-In button
   - **Phone**: Enter a phone number (format: `+91` + 10 digits)

#### Testing Google OAuth:
1. Click the Google Sign-In button
2. Select your Google account
3. Approve the permissions
4. You should be logged in and see the FloodSafe home screen

#### Testing Phone Auth:
1. Switch to the **Phone** tab
2. Enter your phone number (e.g., `9876543210`)
3. Click **Send OTP**
4. Check your phone for the SMS with the 6-digit code
5. Enter the OTP code
6. You should be logged in

---

## Troubleshooting

### Google OAuth Issues

**Problem**: "Unauthorized JavaScript origin"
- **Solution**: Make sure you added `http://localhost:5175` to Authorized JavaScript origins in Google Cloud Console

**Problem**: "Access blocked: This app's request is invalid"
- **Solution**: Complete the OAuth consent screen configuration in Step 7

**Problem**: "redirect_uri_mismatch"
- **Solution**: Add the exact redirect URI to Google Cloud Console credentials

### Firebase Phone Auth Issues

**Problem**: "reCAPTCHA not working"
- **Solution**: Make sure `localhost` is in your Firebase Authorized domains

**Problem**: "SMS not received"
- **Solution**:
  - Check that Phone authentication is enabled in Firebase Console
  - Verify the phone number format: `+91` followed by 10 digits
  - Check Firebase usage limits (free tier: 10K/month)

**Problem**: "Firebase not configured"
- **Solution**: Make sure all `VITE_FIREBASE_*` variables are set in frontend `.env`

### Database Issues

**Problem**: Migration fails
- **Solution**:
  - Check that PostgreSQL is running
  - Verify DATABASE_URL in backend `.env`
  - Make sure the database exists

---

## Security Notes for Production

When deploying to production:

1. **Generate a strong JWT secret**:
   ```bash
   # Generate a secure random key (32+ characters)
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Update Google OAuth**:
   - Add your production domain to Authorized JavaScript origins
   - Add your production domain to Authorized redirect URIs
   - Publish your OAuth consent screen

3. **Update Firebase**:
   - Add your production domain to Authorized domains
   - Consider upgrading to Blaze plan for production usage

4. **Use HTTPS**: Authentication requires HTTPS in production

---

## Quick Reference: What Goes Where

| Configuration | Location | Value |
|---------------|----------|-------|
| Google Client ID | Backend `.env` → `GOOGLE_CLIENT_ID` | `123...apps.googleusercontent.com` |
| Google Client ID | Frontend `.env` → `VITE_GOOGLE_CLIENT_ID` | Same as above |
| Google Client Secret | Backend `.env` → `GOOGLE_CLIENT_SECRET` | `GOCSPX-...` |
| Firebase Project ID | Backend `.env` → `FIREBASE_PROJECT_ID` | `gen-lang-client-0669818939` |
| Firebase API Key | Frontend `.env` → `VITE_FIREBASE_API_KEY` | `AIza...` |
| Firebase Auth Domain | Frontend `.env` → `VITE_FIREBASE_AUTH_DOMAIN` | `gen-lang-client-0669818939.firebaseapp.com` |
| All other Firebase config | Frontend `.env` → `VITE_FIREBASE_*` | From Firebase console |

---

## Need Help?

If you encounter any issues:
1. Check the browser console for error messages
2. Check the backend logs in Docker
3. Verify all environment variables are set correctly
4. Ensure Firebase and Google Cloud configurations match the guide

---

**Your authentication system is now ready! 🎉**

Users can now sign in with:
- ✅ Google Account (one-click)
- ✅ Phone Number + OTP (SMS verification)

All using **FREE** services!
