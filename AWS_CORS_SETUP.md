# AWS API Gateway CORS Configuration Guide

## Problem
Your browser is blocking API requests because CORS (Cross-Origin Resource Sharing) is not configured on your API Gateway. This is a security feature that prevents unauthorized cross-origin requests.

## Solution: Enable CORS on API Gateway

### Method 1: Using AWS Console (Easiest)

#### Step 1: Open API Gateway Console
1. Go to AWS Console: https://console.aws.amazon.com/apigateway
2. Select your API: `agrichain-api` (or whatever your API is named)

#### Step 2: Enable CORS for Each Resource

For **EACH endpoint** (`/notify`, `/simulate`, `/predict`, `/supply`, `/validate`):

1. Click on the resource (e.g., `/notify`)
2. Click **Actions** dropdown
3. Select **Enable CORS**
4. Configure CORS settings:
   ```
   Access-Control-Allow-Origin: *
   Access-Control-Allow-Headers: Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token
   Access-Control-Allow-Methods: GET,POST,PUT,DELETE,OPTIONS
   ```
5. Click **Enable CORS and replace existing CORS headers**
6. Click **Yes, replace existing values**

#### Step 3: Deploy API
1. Click **Actions** dropdown
2. Select **Deploy API**
3. Choose deployment stage (or create new stage like `prod`)
4. Click **Deploy**

### Method 2: Add CORS Headers in Lambda Functions

Add CORS headers to each Lambda function's response:

#### Update forecast/handler.py:
```python
def handler(event, context):
    # ... your existing code ...
    
    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
        },
        "body": json.dumps({
            "forecast_price": float(prediction[0])
        })
    }
```

#### Update supply/handler.py:
```python
def handler(event, context):
    # ... your existing code ...
    
    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
        },
        "body": json.dumps({"message": "Supply recorded"})
    }
```

#### Update simulation/handler.py:
```python
def handler(event, context):
    # ... your existing code ...
    
    return {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
        },
        "body": json.dumps({
            "total_supply": total,
            "estimated_bonus": bulk_bonus
        })
    }
```

### Method 3: Using AWS CLI

```bash
# For each resource, enable CORS
aws apigateway put-integration-response \
    --rest-api-id YOUR_API_ID \
    --resource-id YOUR_RESOURCE_ID \
    --http-method OPTIONS \
    --status-code 200 \
    --response-parameters '{"method.response.header.Access-Control-Allow-Headers":"'"'"'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token'"'"'","method.response.header.Access-Control-Allow-Methods":"'"'"'GET,POST,PUT,DELETE,OPTIONS'"'"'","method.response.header.Access-Control-Allow-Origin":"'"'"'*'"'"'"}'
```

## Verification

### Test CORS Configuration

1. Open browser DevTools (F12)
2. Go to Console tab
3. Run this command:
```javascript
fetch('https://1nz6i4er02.execute-api.us-east-1.amazonaws.com/notify?phoneNumber=test&crop=Tomato&mandi=Kolar')
  .then(response => response.json())
  .then(data => console.log('Success:', data))
  .catch(error => console.error('Error:', error));
```

### Expected Results

**Before CORS:**
```
Access to fetch at 'https://...' from origin 'http://localhost:5173' 
has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header 
is present on the requested resource.
```

**After CORS:**
```
Success: { ... response data ... }
```

## Common CORS Errors and Solutions

### Error 1: "No 'Access-Control-Allow-Origin' header"
**Solution**: Enable CORS on API Gateway or add headers in Lambda

### Error 2: "Method not allowed"
**Solution**: Add OPTIONS method to your API Gateway resources

### Error 3: "Preflight request failed"
**Solution**: Ensure OPTIONS method returns 200 with proper headers

## Security Note

Using `Access-Control-Allow-Origin: *` allows requests from any domain. For production:

1. Replace `*` with your specific domain:
   ```
   Access-Control-Allow-Origin: https://yourdomain.com
   ```

2. Or use multiple domains:
   ```python
   allowed_origins = [
       'https://yourdomain.com',
       'https://www.yourdomain.com',
       'http://localhost:5173'  # for development
   ]
   
   origin = event['headers'].get('origin', '')
   if origin in allowed_origins:
       headers['Access-Control-Allow-Origin'] = origin
   ```

## Quick Fix for Testing

If you need to test immediately without configuring CORS:

### Option 1: Use Browser Extension
Install a CORS browser extension:
- Chrome: "CORS Unblock" or "Allow CORS"
- Firefox: "CORS Everywhere"

**Warning**: Only use for development/testing!

### Option 2: Use Proxy
Run a local proxy that adds CORS headers:

```bash
npm install -g local-cors-proxy
lcp --proxyUrl https://1nz6i4er02.execute-api.us-east-1.amazonaws.com
```

Then update `.env`:
```
VITE_API_URL=http://localhost:8010/proxy
```

## Recommended Approach

**For immediate testing**: Use Method 2 (add headers in Lambda functions)  
**For production**: Use Method 1 (configure CORS in API Gateway)

## After Fixing CORS

Once CORS is configured:
1. Redeploy your Lambda functions (if you modified them)
2. Deploy your API Gateway
3. Wait 1-2 minutes for changes to propagate
4. Test your frontend application
5. Check browser console for any remaining errors

---

**Need Help?**
- Check AWS CloudWatch logs for Lambda errors
- Use browser DevTools Network tab to see request/response
- Verify API Gateway deployment stage is correct