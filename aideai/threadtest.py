subject = "Test Failure - Sample <b>{sample}</b> in Project <b>{project}</b>"
message = """Hi,

I regret to inform you that an error has occurred in the project named project_name. The application encountered an unexpected exception, causing the process to halt.

Exception Details:
{response}

This exception has impacted the project's functionality, and it requires immediate attention. Please review the exception details and take appropriate actions to resolve the issue as soon as possible.

If you require any assistance or further information regarding the error, please let me know. I'm here to help.

Thank you for your prompt attention to this matter."""

print(subject.strip())
print(message.strip())